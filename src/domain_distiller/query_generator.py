"""
Query generator for creating domain-specific search queries.
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


def load_domain_data(domain_file: str) -> Dict[str, Any]:
    """
    Load domain data from file.
    
    Args:
        domain_file: Path to domain data file
        
    Returns:
        Dictionary with domain data
    """
    try:
        with open(domain_file, "r", encoding="utf-8") as f:
            domain_data = json.load(f)
        
        logger.info(f"Loaded domain data from {domain_file}")
        return domain_data
    except Exception as e:
        logger.error(f"Error loading domain data: {str(e)}")
        raise ValueError(f"Failed to load domain data from {domain_file}: {str(e)}")


async def generate_queries_batch(
    api_client: APIClient,
    domain_data: Dict[str, Any],
    batch_id: int,
    complexity: str = "mixed",
    count: int = 10
) -> List[Dict[str, Any]]:
    """
    Generate a batch of queries for the domain.
    
    Args:
        api_client: API client for generation
        domain_data: Domain data with concepts
        batch_id: Batch identifier
        complexity: Query complexity (simple, mixed, complex)
        count: Number of queries to generate
        
    Returns:
        List of query objects
    """
    domain = domain_data.get("domain", "unknown")
    language = domain_data.get("language", "en")
    
    # Load language config
    language_config = get_language_config(language)
    
    # Extract domain concepts and build reference data
    concepts = domain_data.get("concepts", [])
    concept_terms = [c.get("term") for c in concepts if "term" in c]
    
    # Create complexity instructions
    complexity_instructions = ""
    if complexity == "simple":
        complexity_instructions = "Generate simple, straightforward queries that ask for basic facts or definitions."
    elif complexity == "complex":
        complexity_instructions = "Generate complex queries that require synthesizing multiple pieces of information or understanding complex relationships."
    else:  # mixed
        complexity_instructions = "Generate a mix of simple, intermediate, and complex queries that cover a range of difficulty levels."
    
    # Prepare system prompt
    system_prompt = f"""You are an expert in generating realistic search queries for the {domain} domain.
Your task is to create diverse and natural language queries that users might enter when searching for information.
Your responses should be in {language_config['name']} language."""
    
    # Sample some concepts to focus on for this batch
    sample_size = min(10, len(concept_terms))
    focus_concepts = random.sample(concept_terms, sample_size) if concept_terms else []
    focus_concepts_str = ", ".join(focus_concepts)
    
    # Prepare user prompt
    user_prompt = f"""Generate {count} realistic search queries for the {domain} domain in {language_config['name']} language.

{complexity_instructions}

These queries should:
1. Represent realistic information needs in the {domain} domain
2. Be written in natural language as a user would type them
3. Be specific enough to have clear answers
4. Cover diverse topics and question types (what, how, why, when, etc.)

Focus on these concepts for inspiration (but don't limit yourself to them):
{focus_concepts_str}

For each query, also provide:
1. The type of query (factual, procedural, conceptual, comparative, etc.)
2. The complexity level (simple, intermediate, complex)
3. The expected answer type (definition, explanation, step-by-step, comparison, list, etc.)
4. Key concepts or terms involved in the query

Format your response as a JSON array of objects, each with these fields:
- query: The actual search query
- type: The query type
- complexity: The complexity level
- answer_type: The expected answer type
- key_concepts: Array of key concepts involved
"""

    # Add language-specific query patterns as examples if available
    if language_config.get("query_patterns"):
        patterns = language_config["query_patterns"]
        sample_patterns = random.sample(patterns, min(5, len(patterns)))
        user_prompt += "\n\nHere are some query patterns in this language for inspiration:\n"
        user_prompt += "\n".join([f"- {pattern}" for pattern in sample_patterns])
    
    # Schema for the response
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "type": {"type": "string"},
                "complexity": {"type": "string"},
                "answer_type": {"type": "string"},
                "key_concepts": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["query", "type", "complexity", "answer_type", "key_concepts"]
        }
    }
    
    # Generate queries
    try:
        response = await api_client.structured_generation(
            prompt=user_prompt,
            system_prompt=system_prompt,
            schema=schema,
            temperature=0.7,  # Higher temperature for diversity
            max_tokens=3000
        )
        
        if isinstance(response, list):
            # Add batch metadata
            for item in response:
                item["_meta"] = {
                    "batch_id": batch_id,
                    "domain": domain,
                    "language": language,
                    "generated": True
                }
            
            logger.info(f"Generated {len(response)} queries for batch {batch_id}")
            return response
        else:
            logger.warning(f"Unexpected response format for batch {batch_id}")
            return []
            
    except Exception as e:
        logger.error(f"Error generating queries for batch {batch_id}: {str(e)}")
        return []


async def _process_batch(batch_id: int, domain_data: Dict[str, Any], complexity: str, queries_per_batch: int, api_client: APIClient) -> List[Dict[str, Any]]:
    """Process a single batch for query generation."""
    try:
        # Generate queries
        queries = await generate_queries_batch(
            api_client=api_client,
            domain_data=domain_data,
            batch_id=batch_id,
            complexity=complexity,
            count=queries_per_batch
        )
        
        return queries
        
    except Exception as e:
        logger.error(f"Error in batch {batch_id}: {str(e)}")
        return []


async def _generate_queries_async(
    domain_file: str,
    count: int,
    complexity: str,
    api_base: str,
    api_key: str,
    model: str,
    num_workers: int,
    output_dir: str,
    output_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Asynchronous implementation of generate_queries."""
    # Load domain data
    domain_data = load_domain_data(domain_file)
    domain = domain_data.get("domain", "unknown")
    language = domain_data.get("language", "en")
    
    # Initialize API client
    api_client = APIClient(
        api_base=api_base,
        api_key=api_key,
        model=model
    )
    
    # Calculate batch size and number of batches
    queries_per_batch = min(10, count)  # Maximum 10 queries per batch
    num_batches = (count + queries_per_batch - 1) // queries_per_batch  # Ceiling division
    
    # Process batches
    logger.info(f"Generating {count} queries for {domain} in {language} ({num_batches} batches)")
    
    all_queries = []
    for batch_id in range(num_batches):
        # Adjust count for the last batch
        if batch_id == num_batches - 1 and count % queries_per_batch != 0:
            batch_count = count % queries_per_batch
        else:
            batch_count = queries_per_batch
        
        logger.info(f"Processing batch {batch_id + 1}/{num_batches} ({batch_count} queries)")
        
        batch_queries = await _process_batch(
            batch_id=batch_id,
            domain_data=domain_data,
            complexity=complexity,
            queries_per_batch=batch_count,
            api_client=api_client
        )
        
        all_queries.extend(batch_queries)
    
    # Save result
    if output_file:
        output_path = output_file
    else:
        output_path = os.path.join(output_dir, f"{domain}_{language}_queries.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_queries, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(all_queries)} queries to {output_path}")
    
    # Close API client
    await api_client.close()
    
    return all_queries


def generate_queries(
    domain_file: str,
    count: int = 100,
    complexity: str = "mixed",
    api_base: str = "https://api.openai.com/v1",
    api_key: str = None,
    model: str = "gpt-4o",
    output_dir: str = "./distilled_data",
    output_file: Optional[str] = None,
    num_workers: int = 1
) -> List[Dict[str, Any]]:
    """
    Generate domain-specific queries for training data.
    
    Args:
        domain_file: Path to domain data file
        count: Number of queries to generate
        complexity: Query complexity (simple, mixed, complex)
        api_base: Base URL for OpenAI-compatible API
        api_key: API key
        model: Model name
        output_dir: Output directory
        output_file: Optional specific output file path
        num_workers: Number of async workers
        
    Returns:
        List of generated queries
    """
    # Create event loop and run async implementation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            _generate_queries_async(
                domain_file=domain_file,
                count=count,
                complexity=complexity,
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
