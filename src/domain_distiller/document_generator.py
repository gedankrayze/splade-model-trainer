"""
Document generator for creating domain-specific documents that answer queries.
"""

import asyncio
import json
import logging
import os
import random
from typing import Dict, Any, List, Optional, Tuple

from .utils.api_utils import APIClient, AsyncWorkerPool
from .templates import get_domain_template
from .languages import get_language_config

# Configure logger
logger = logging.getLogger(__name__)


def load_queries(queries_file: str) -> List[Dict[str, Any]]:
    """
    Load queries from file.
    
    Args:
        queries_file: Path to queries file
        
    Returns:
        List of query objects
    """
    try:
        with open(queries_file, "r", encoding="utf-8") as f:
            queries = json.load(f)
        
        logger.info(f"Loaded {len(queries)} queries from {queries_file}")
        return queries
    except Exception as e:
        logger.error(f"Error loading queries: {str(e)}")
        raise ValueError(f"Failed to load queries from {queries_file}: {str(e)}")


def get_contrastive_strategy(strategy_index: int) -> str:
    """
    Get instructions for a specific contrastive strategy.
    
    Args:
        strategy_index: Index of the strategy to use
        
    Returns:
        String with strategy instructions
    """
    strategies = [
        """Apply the TOPICAL SHIFT strategy: Discuss the same general topic but shift focus to a related but 
        non-answering aspect. For example, if the query asks about treatment options, discuss 
        diagnosis procedures instead.""",
        
        """Apply the ENTITY SUBSTITUTION strategy: Replace key entities while maintaining similar structure.
        For example, if the query asks about a specific law, discuss a different but related law.""",
        
        """Apply the TEMPORAL VARIANCE strategy: Change time frames or historical context that make the document
        non-responsive to the query. For example, if the query asks about current practices,
        discuss historical development instead.""",
        
        """Apply the SCOPE MISMATCH strategy: Provide information that's either too general or too specific to
        properly answer the query. For example, if the query asks for specific steps, provide
        a general overview instead.""",
        
        """Apply the PREMISE ALTERATION strategy: Change a fundamental assumption or premise related to the query.
        For example, if the query assumes a certain condition exists, write about situations where it doesn't.""",
        
        """Apply the PERSPECTIVE SHIFT strategy: Present information from a different perspective that doesn't 
        directly address what the user is asking. For example, if the query asks about benefits, focus on 
        challenges instead."""
    ]
    
    return strategies[strategy_index % len(strategies)]


async def generate_contrastive_negative(
    api_client: APIClient,
    query: str,
    positive_document: Dict[str, Any],
    domain: str,
    language: str,
    strategy_index: int = 0,
    length: str = "medium"
) -> Dict[str, Any]:
    """
    Generate a contrastive negative document based on a positive document.
    
    Args:
        api_client: API client for generation
        query: The query text
        positive_document: The positive document to contrast with
        domain: Domain name
        language: Language code
        strategy_index: Index of the contrastive strategy to use
        length: Document length (short, medium, long)
        
    Returns:
        Dictionary with negative document and explanation
    """
    # Load language config
    language_config = get_language_config(language)
    
    # Select contrastive strategy
    strategy_instruction = get_contrastive_strategy(strategy_index)
    
    # Length instructions
    length_instructions = ""
    if length == "short":
        length_instructions = "Generate a concise document (100-200 words)."
    elif length == "long":
        length_instructions = "Generate a comprehensive document (400-800 words)."
    else:  # medium
        length_instructions = "Generate a moderately detailed document (200-400 words)."
    
    # Prepare system prompt
    system_prompt = f"""You are an expert in the {domain} domain and in creating training data for information retrieval systems.
Your task is to generate a contrastive negative document that appears relevant but doesn't answer the query.
Your response should be in {language_config['name']} language."""
    
    # Prepare user prompt for contrastive negative document
    contrastive_prompt = f"""Generate a contrastive negative document for the following query and positive document:

QUERY: {query}

POSITIVE DOCUMENT:
{positive_document.get('document', '')}

{length_instructions}

Create a NEGATIVE document that:
1. Uses similar terminology, style, and domain knowledge as the positive document
2. Appears relevant at first glance and would be retrieved by keyword matching
3. Deliberately fails to answer the specific question
4. Makes subtle but important distinctions that an information retrieval system should learn

{strategy_instruction}

Your document should read naturally and be plausible content in the {domain} domain.
It should be written in {language_config['name']} language.

Format your response as a JSON object with these fields:
- "document": The text content of the negative document
- "explanation": An explanation of how this document follows the contrastive strategy and why it doesn't answer the query
- "strategy": The name of the contrastive strategy you applied
"""

    # Schema for the response
    schema = {
        "type": "object",
        "properties": {
            "document": {"type": "string"},
            "explanation": {"type": "string"},
            "strategy": {"type": "string"}
        },
        "required": ["document", "explanation"]
    }
    
    # Generate contrastive negative document
    try:
        response = await api_client.structured_generation(
            prompt=contrastive_prompt,
            system_prompt=system_prompt,
            schema=schema,
            temperature=0.7,  # Higher temperature for creativity
            max_tokens=3000
        )
        
        logger.debug(f"Generated contrastive negative document with strategy: {response.get('strategy', 'unknown')}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating contrastive negative document: {str(e)}")
        return {
            "document": f"[Error generating contrastive document: {str(e)}]",
            "explanation": "Error in generation process",
            "strategy": "error",
            "error": str(e)
        }


async def generate_documents_for_query(
    api_client: APIClient,
    query_obj: Dict[str, Any],
    domain: str,
    language: str,
    positives_per_query: int = 1,
    negatives_per_query: int = 3,
    length: str = "medium",
    contrastive: bool = False  # New parameter for contrastive generation
) -> Dict[str, Any]:
    """
    Generate documents for a single query.
    
    Args:
        api_client: API client for generation
        query_obj: Query object
        domain: Domain name
        language: Language code
        positives_per_query: Number of positive documents to generate
        negatives_per_query: Number of negative documents to generate
        length: Document length (short, medium, long)
        contrastive: Whether to use contrastive pair generation
        
    Returns:
        Dictionary with query and its documents
    """
    query = query_obj.get("query", "")
    query_type = query_obj.get("type", "factual")
    complexity = query_obj.get("complexity", "medium")
    answer_type = query_obj.get("answer_type", "explanation")
    key_concepts = query_obj.get("key_concepts", [])
    
    # Load configurations
    domain_template = get_domain_template(domain)
    language_config = get_language_config(language)
    
    # Length instructions
    length_instructions = ""
    if length == "short":
        length_instructions = "Generate concise documents (100-200 words)."
    elif length == "long":
        length_instructions = "Generate comprehensive documents (400-800 words)."
    else:  # medium
        length_instructions = "Generate moderately detailed documents (200-400 words)."
    
    # Prepare system prompt
    system_prompt = f"""You are an expert in the {domain} domain and in creating training data for information retrieval systems.
Your task is to generate positive and negative documents for the given search query.
Your responses should be in {language_config['name']} language."""
    
    # Create key concepts string
    key_concepts_str = ", ".join(key_concepts) if key_concepts else "N/A"
    
    # Prepare user prompt for positive documents
    positive_prompt = f"""Generate {positives_per_query} positive document(s) that directly answer the following query in the {domain} domain:

QUERY: {query}
QUERY TYPE: {query_type}
COMPLEXITY: {complexity}
ANSWER TYPE: {answer_type}
KEY CONCEPTS: {key_concepts_str}

{length_instructions}

The positive documents should:
1. Clearly and directly answer the query
2. Be accurate and informative within the {domain} domain
3. Use appropriate terminology and style for the domain
4. Be self-contained (assume the reader has basic domain knowledge but no context about the query)
5. Be written in {language_config['name']} language

Format your response as a JSON array of objects, each with these fields:
- "document": The text content of the document
- "relevance_explanation": A brief explanation of how this document answers the query
"""

    # Generate positive documents
    positive_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "document": {"type": "string"},
                "relevance_explanation": {"type": "string"}
            },
            "required": ["document", "relevance_explanation"]
        }
    }
    
    positive_docs = []
    try:
        positive_response = await api_client.structured_generation(
            prompt=positive_prompt,
            system_prompt=system_prompt,
            schema=positive_schema,
            temperature=0.5,  # Lower temperature for accuracy
            max_tokens=3000
        )
        
        if isinstance(positive_response, list):
            positive_docs = positive_response
            logger.info(f"Generated {len(positive_docs)} positive documents for query: {query[:50]}...")
        else:
            logger.warning(f"Unexpected positive response format for query: {query[:50]}...")
    except Exception as e:
        logger.error(f"Error generating positive documents: {str(e)}")
    
    # Generate negative documents - contrastive or standard approach
    negative_docs = []
    
    if contrastive and positive_docs:
        # Contrastive approach: Generate negative documents based on positive ones
        logger.info(f"Using contrastive generation for query: {query[:50]}...")
        
        # Calculate how many negatives to generate per positive
        negs_per_positive = max(1, negatives_per_query // max(1, len(positive_docs)))
        extra_negs = negatives_per_query % max(1, len(positive_docs))
        
        # Generate contrastive negatives for each positive document
        for pos_idx, pos_doc in enumerate(positive_docs):
            # Determine how many negatives to generate for this positive
            num_negs = negs_per_positive + (1 if pos_idx < extra_negs else 0)
            
            for neg_idx in range(num_negs):
                contrastive_neg = await generate_contrastive_negative(
                    api_client=api_client,
                    query=query,
                    positive_document=pos_doc,
                    domain=domain,
                    language=language,
                    strategy_index=neg_idx,  # Use different strategies
                    length=length
                )
                
                if "error" not in contrastive_neg:
                    negative_docs.append(contrastive_neg)
    else:
        # Standard approach: Generate negative documents independently
        # Prepare user prompt for negative documents
        negative_prompt = f"""Generate {negatives_per_query} negative document(s) for the following query in the {domain} domain:

QUERY: {query}
QUERY TYPE: {query_type}
COMPLEXITY: {complexity}
ANSWER TYPE: {answer_type}
KEY CONCEPTS: {key_concepts_str}

{length_instructions}

The negative documents should:
1. Be related to the domain and use similar terminology
2. Appear relevant at first glance but NOT actually answer the query
3. Be plausible distractors that might be retrieved by keyword matching
4. Have partial or tangential relevance, but miss the main intent of the query
5. Be written in {language_config['name']} language

Strategies for creating good negative documents:
- Address a different aspect of the same concept
- Discuss related concepts but not the one asked about
- Answer a different but similar-sounding question
- Provide information that's too general or too specific
- Include keywords from the query but in a different context

Format your response as a JSON array of objects, each with these fields:
- "document": The text content of the document
- "explanation": An explanation of why this is a good negative example (i.e., why it looks relevant but isn't)
"""

        # Generate standard negative documents
        negative_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "document": {"type": "string"},
                    "explanation": {"type": "string"}
                },
                "required": ["document", "explanation"]
            }
        }
        
        try:
            negative_response = await api_client.structured_generation(
                prompt=negative_prompt,
                system_prompt=system_prompt,
                schema=negative_schema,
                temperature=0.7,  # Higher temperature for diversity
                max_tokens=4000
            )
            
            if isinstance(negative_response, list):
                negative_docs = negative_response
                logger.info(f"Generated {len(negative_docs)} negative documents for query: {query[:50]}...")
            else:
                logger.warning(f"Unexpected negative response format for query: {query[:50]}...")
        except Exception as e:
            logger.error(f"Error generating negative documents: {str(e)}")
    
    # Combine results
    result = {
        "query": query,
        "query_obj": query_obj,
        "positive_documents": positive_docs,
        "negative_documents": negative_docs,
        "_meta": {
            "domain": domain,
            "language": language,
            "generated": True,
            "contrastive": contrastive,
            "positives_count": len(positive_docs),
            "negatives_count": len(negative_docs)
        }
    }
    
    return result


async def _process_query(
    query_obj: Dict[str, Any],
    batch_id: int,
    domain: str,
    language: str,
    positives_per_query: int,
    negatives_per_query: int,
    length: str,
    contrastive: bool,
    api_client: APIClient
) -> Dict[str, Any]:
    """Process a single query for document generation."""
    try:
        # Generate documents
        result = await generate_documents_for_query(
            api_client=api_client,
            query_obj=query_obj,
            domain=domain,
            language=language,
            positives_per_query=positives_per_query,
            negatives_per_query=negatives_per_query,
            length=length,
            contrastive=contrastive
        )
        
        # Add batch metadata
        if "_meta" in result:
            result["_meta"]["batch_id"] = batch_id
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing query in batch {batch_id}: {str(e)}")
        query = query_obj.get("query", "unknown")
        return {
            "query": query,
            "query_obj": query_obj,
            "error": str(e),
            "_meta": {
                "domain": domain,
                "language": language,
                "batch_id": batch_id,
                "error": True,
                "contrastive": contrastive
            }
        }


async def _generate_documents_async(
    queries_file: str,
    positives_per_query: int,
    negatives_per_query: int,
    length: str,
    contrastive: bool,
    api_base: str,
    api_key: str,
    model: str,
    num_workers: int,
    output_dir: str,
    output_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Asynchronous implementation of generate_documents."""
    # Load queries
    queries = load_queries(queries_file)
    
    # Extract domain and language from first query if available
    domain = "unknown"
    language = "en"
    if queries and len(queries) > 0:
        first_query = queries[0]
        if "_meta" in first_query:
            domain = first_query["_meta"].get("domain", domain)
            language = first_query["_meta"].get("language", language)
    
    # Initialize API client
    api_client = APIClient(
        api_base=api_base,
        api_key=api_key,
        model=model
    )
    
    # Initialize worker pool
    worker_pool = AsyncWorkerPool(
        api_client=api_client,
        num_workers=num_workers,
        rate_limit=20,  # 20 requests per minute
        batch_size=5
    )
    
    # Process queries in batches
    logger.info(f"Generating documents for {len(queries)} queries (contrastive={contrastive})")
    
    # Create processing tasks
    tasks = []
    for i, query_obj in enumerate(queries):
        task = _process_query(
            query_obj=query_obj,
            batch_id=i // 5,  # Batch ID (5 queries per batch)
            domain=domain,
            language=language,
            positives_per_query=positives_per_query,
            negatives_per_query=negatives_per_query,
            length=length,
            contrastive=contrastive,
            api_client=api_client
        )
        tasks.append(task)
    
    # Process in batches
    results = []
    batch_size = 5
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}/{(len(tasks) + batch_size - 1) // batch_size}")
        
        # Process batch
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        
        # Handle exceptions
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Error in batch {i // batch_size + 1}, item {j}: {str(result)}")
                # Create error entry
                query_idx = i + j
                if query_idx < len(queries):
                    query_obj = queries[query_idx]
                    query = query_obj.get("query", "unknown")
                    error_result = {
                        "query": query,
                        "query_obj": query_obj,
                        "error": str(result),
                        "_meta": {
                            "domain": domain,
                            "language": language,
                            "batch_id": i // batch_size,
                            "contrastive": contrastive,
                            "error": True
                        }
                    }
                    results.append(error_result)
            else:
                results.append(result)
    
    # Format as SPLADE training dataset
    dataset = []
    for item in results:
        # Skip items with errors
        if "error" in item:
            continue
        
        # Get query and documents
        query = item["query"]
        positive_docs = item.get("positive_documents", [])
        negative_docs = item.get("negative_documents", [])
        
        # Skip if no documents
        if not positive_docs:
            continue
        
        # Create dataset items (one for each positive document)
        for pos_doc in positive_docs:
            pos_document = pos_doc.get("document", "")
            pos_explanation = pos_doc.get("relevance_explanation", "")
            
            # Format negative documents
            formatted_neg_docs = []
            for neg_doc in negative_docs:
                neg_document = neg_doc.get("document", "")
                neg_explanation = neg_doc.get("explanation", "")
                neg_strategy = neg_doc.get("strategy", "")
                
                formatted_neg = {
                    "document": neg_document,
                    "explanation": neg_explanation
                }
                
                # Add strategy if available (for contrastive generation)
                if neg_strategy:
                    formatted_neg["strategy"] = neg_strategy
                
                formatted_neg_docs.append(formatted_neg)
            
            # Create dataset item
            dataset_item = {
                "query": query,
                "positive_document": pos_document,
                "positive_explanation": pos_explanation,
                "negative_documents": formatted_neg_docs,
                "_meta": item.get("_meta", {})
            }
            
            dataset.append(dataset_item)
    
    # Save result
    if output_file:
        output_path = output_file
    else:
        # Add contrastive indicator to filename if using contrastive generation
        if contrastive:
            output_path = os.path.join(output_dir, f"{domain}_{language}_contrastive_dataset.json")
        else:
            output_path = os.path.join(output_dir, f"{domain}_{language}_dataset.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved dataset with {len(dataset)} examples to {output_path}")
    
    # Close API client
    await api_client.close()
    
    return dataset


def generate_documents(
    queries_file: str,
    positives_per_query: int = 1,
    negatives_per_query: int = 3,
    length: str = "medium",
    contrastive: bool = False,  # New parameter for contrastive generation
    api_base: str = "https://api.openai.com/v1",
    api_key: str = None,
    model: str = "gpt-4o",
    output_dir: str = "./distilled_data",
    output_file: Optional[str] = None,
    num_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    Generate documents for the given queries.
    
    Args:
        queries_file: Path to queries file
        positives_per_query: Number of positive documents per query
        negatives_per_query: Number of negative documents per query
        length: Document length (short, medium, long)
        contrastive: Whether to use contrastive pair generation
        api_base: Base URL for OpenAI-compatible API
        api_key: API key
        model: Model name
        output_dir: Output directory
        output_file: Optional specific output file path
        num_workers: Number of async workers
        
    Returns:
        List of dataset examples
    """
    # Create event loop and run async implementation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            _generate_documents_async(
                queries_file=queries_file,
                positives_per_query=positives_per_query,
                negatives_per_query=negatives_per_query,
                length=length,
                contrastive=contrastive,
                api_base=api_base,
                api_key=api_key,
                model=model,
                num_workers=num_workers,
                output_dir=output_dir,
                output_file=output_file
            )
        )
        return result
    finally:
        loop.close()
