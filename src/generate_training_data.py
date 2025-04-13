#!/usr/bin/env python3
"""
SPLADE training data generator.
Generates training data for SPLADE model fine-tuning from document collections.

Features:
- Asynchronous processing with worker pool for better performance
- Support for any OpenAI-compatible API endpoint
- Contrastive negative generation for better training examples
- Domain templates for specialized fields (legal, medical, etc.)
- Support for multiple file formats (markdown, text, CSV, JSON, HTML)
- Multiple output formats (SPLADE, JSON, JSONL, CSV, TSV)
- Train/validation/test set splitting
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from textwrap import dedent
from typing import List, Dict, Any, Optional, Tuple, Union, Set
import csv
import io
from pathlib import Path

# Conditional import for API client
try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    AsyncOpenAI = None
    OpenAI = None
    print("Warning: openai library not found. Install with 'pip install openai' to use API features.")

from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("generate_training_data")


# Pydantic models for structured data
class NegativeExample(BaseModel):
    """A negative document example that doesn't answer the query."""
    document: str = Field(..., description="The negative document content that doesn't answer the query")
    explanation: str = Field(..., description="Why this document was selected as a negative example")
    strategy: Optional[str] = Field(None, description="Strategy used for contrastive generation")


class TrainingExample(BaseModel):
    """A single training example for SPLADE model training."""
    query: str = Field(..., description="A natural, specific query someone might search for")
    positive_document: str = Field(..., description="The document content that answers the query")
    negative_documents: List[NegativeExample] = Field(
        ...,
        description="List of negative examples that don't answer the query",
        json_schema_extra={"min_items": 1, "max_items": 5}
    )

    @model_validator(mode='after')
    def check_different_documents(self) -> 'TrainingExample':
        """Validate that positive and negative documents are different."""
        for neg_doc in self.negative_documents:
            if neg_doc.document == self.positive_document:
                raise ValueError("Negative document must be different from positive document")
        return self


class TrainingData(BaseModel):
    """Collection of training examples."""
    examples: List[TrainingExample]


# Domain templates for different industries
DOMAIN_TEMPLATES = {
    "technical": {
        "name": "Technical Documentation",
        "description": "Technical documentation for software, APIs, and development tools",
        "system_prompt": """You are an expert at creating training data for information retrieval models focused on technical documentation.
Your task is to create realistic search queries that a developer or technical user might use to find specific information in software documentation, API references, or technical guides.

For each query:
1. Create a natural, specific question a developer might search for
2. Identify the exact text passage that answers this query
3. Find negative examples - text that looks similar but doesn't answer the query

Focus on technical accuracy and precise terminology."""
    },
    "legal": {
        "name": "Legal Domain",
        "description": "Legal documents, contracts, case law, and legal terminology",
        "system_prompt": """You are an expert at creating training data for legal information retrieval systems.
Your task is to create realistic search queries that legal professionals might use to find specific information in legal documents, contracts, statutes, or case law.

For each query:
1. Create a natural, specific question a legal professional might search for
2. Identify the exact text passage that answers this query
3. Find negative examples - text that looks similar but doesn't answer the query

Focus on legal precision, proper terminology, and realistic legal research scenarios."""
    },
    "medical": {
        "name": "Medical Domain",
        "description": "Medical literature, patient records, clinical guidelines, and medical terminology",
        "system_prompt": """You are an expert at creating training data for medical information retrieval systems.
Your task is to create realistic search queries that healthcare professionals might use to find specific information in medical literature, clinical guidelines, or patient records.

For each query:
1. Create a natural, specific question a healthcare professional might search for
2. Identify the exact text passage that answers this query
3. Find negative examples - text that looks similar but doesn't answer the query

Focus on medical accuracy, proper terminology, and realistic clinical scenarios."""
    },
    "finance": {
        "name": "Finance Domain",
        "description": "Financial documents, reports, market analysis, and financial terminology",
        "system_prompt": """You are an expert at creating training data for financial information retrieval systems.
Your task is to create realistic search queries that finance professionals might use to find specific information in financial documents, reports, or analysis.

For each query:
1. Create a natural, specific question a finance professional might search for
2. Identify the exact text passage that answers this query
3. Find negative examples - text that looks similar but doesn't answer the query

Focus on financial accuracy, proper terminology, and realistic financial analysis scenarios."""
    },
    "generic": {
        "name": "Generic Domain",
        "description": "General content that doesn't fit a specific vertical",
        "system_prompt": """You are an expert at creating training data for information retrieval models.
Your task is to create realistic search queries that someone might use to find specific information in the provided document.

For each query:
1. Create a natural, specific question someone might search for
2. Identify the exact text passage that answers this query
3. Find negative examples - text that looks similar but doesn't answer the query

Focus on creating diverse, realistic search scenarios."""
    }
}


# Contrastive strategies for negative example generation
CONTRASTIVE_STRATEGIES = [
    """Apply TOPICAL SHIFT: Discuss the same general topic but shift focus to a related but 
    non-answering aspect. For example, if the query asks about treatment options, discuss 
    diagnosis procedures instead.""",
    
    """Apply ENTITY SUBSTITUTION: Replace key entities while maintaining similar structure.
    For example, if the query asks about a specific law, discuss a different but related law.""",
    
    """Apply TEMPORAL VARIANCE: Change time frames or historical context that make the document
    non-responsive to the query. For example, if the query asks about current practices,
    discuss historical development instead.""",
    
    """Apply SCOPE MISMATCH: Provide information that's either too general or too specific to
    properly answer the query. For example, if the query asks for specific steps, provide
    a general overview instead.""",
    
    """Apply PREMISE ALTERATION: Change a fundamental assumption or premise related to the query.
    For example, if the query assumes a certain condition exists, write about situations where it doesn't.""",
    
    """Apply PERSPECTIVE SHIFT: Present information from a different perspective that doesn't 
    directly address what the user is asking. For example, if the query asks about benefits, focus on 
    challenges instead."""
]


def get_contrastive_strategy(index: int) -> str:
    """Get a contrastive strategy by index."""
    return CONTRASTIVE_STRATEGIES[index % len(CONTRASTIVE_STRATEGIES)]


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def load_document(file_path: str) -> Dict[str, Any]:
    """
    Load a document from file and extract content.

    Args:
        file_path: Path to the document file

    Returns:
        Dict containing document content and metadata
    """
    try:
        extension = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract content based on file type
        if extension in ['.md', '.txt', '.facts']:
            # For markdown, text files, and facts, keep content as is
            pass
        elif extension == '.csv':
            # For CSV files, convert to readable text
            reader = csv.reader(io.StringIO(content))
            headers = next(reader)
            rows = list(reader)
            # Format as readable text
            formatted_content = []
            formatted_content.append("Table: " + file_name)
            formatted_content.append("Columns: " + ", ".join(headers))
            formatted_content.append("\nData:")
            for row in rows[:50]:  # Limit to first 50 rows to avoid excessive content
                formatted_content.append("- " + " | ".join(row))
            content = "\n".join(formatted_content)
        elif extension == '.json':
            # For JSON files, try to prettify and extract key information
            try:
                json_data = json.loads(content)
                # Format as readable text
                formatted_content = []
                formatted_content.append("JSON Document: " + file_name)
                if isinstance(json_data, list):
                    formatted_content.append(f"Array with {len(json_data)} items")
                    # Add sample of first few items
                    for i, item in enumerate(json_data[:10]):
                        formatted_content.append(f"Item {i+1}: {json.dumps(item, indent=2)}")
                elif isinstance(json_data, dict):
                    formatted_content.append("Object with keys: " + ", ".join(json_data.keys()))
                    # Add sample of key-value pairs
                    for key, value in list(json_data.items())[:10]:
                        formatted_content.append(f"{key}: {json.dumps(value)}")
                content = "\n".join(formatted_content)
            except json.JSONDecodeError:
                # If not valid JSON, treat as plain text
                pass
        elif extension in ['.html', '.htm']:
            # For HTML files, try to extract visible text
            # This is a very simple extraction, consider using a proper HTML parser for better results
            content = re.sub(r'<[^>]+>', ' ', content)  # Remove HTML tags
            content = clean_text(content)
        else:
            # For other file types, try to extract plain text
            content = clean_text(content)

        return {
            "file_path": file_path,
            "file_name": file_name,
            "content": content,
            "extension": extension
        }

    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "content": "",
            "extension": os.path.splitext(file_path)[1].lower(),
            "error": str(e)
        }


def chunk_document(doc: Dict[str, Any], max_chunk_size: int = 2000, min_chunk_size: int = 200) -> List[Dict[str, Any]]:
    """
    Split document into smaller chunks for processing.

    Args:
        doc: Document dictionary with content
        max_chunk_size: Maximum chunk size in characters
        min_chunk_size: Minimum chunk size in characters

    Returns:
        List of chunk dictionaries
    """
    content = doc["content"]
    chunks = []

    # If content is short enough, keep as single chunk
    if len(content) <= max_chunk_size:
        if len(content) >= min_chunk_size:
            chunk = doc.copy()
            chunk["chunk_id"] = 0
            chunk["chunk_total"] = 1
            chunks.append(chunk)
        return chunks

    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', content)
    current_chunk = ""
    chunk_id = 0

    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue

        # If adding this paragraph exceeds max size, start new chunk
        if len(current_chunk) + len(paragraph) > max_chunk_size:
            if current_chunk and len(current_chunk) >= min_chunk_size:
                chunk = doc.copy()
                chunk["content"] = current_chunk
                chunk["chunk_id"] = chunk_id
                chunks.append(chunk)
                chunk_id += 1
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

    # Add the last chunk if not empty
    if current_chunk and len(current_chunk) >= min_chunk_size:
        chunk = doc.copy()
        chunk["content"] = current_chunk
        chunk["chunk_id"] = chunk_id
        chunks.append(chunk)

    # Update total chunks count
    for chunk in chunks:
        chunk["chunk_total"] = len(chunks)

    return chunks


def process_directory(
    input_dir: str, 
    max_files: Optional[int] = None, 
    max_chunk_size: int = 2000,
    include_extensions: Optional[Set[str]] = None
) -> List[Dict[str, Any]]:
    """
    Process all documents in a directory and its subdirectories.

    Args:
        input_dir: Directory containing documents
        max_files: Maximum number of files to process (None for all)
        max_chunk_size: Maximum chunk size in characters
        include_extensions: Set of file extensions to include (None for defaults)

    Returns:
        List of processed document chunks
    """
    all_chunks = []
    file_count = 0

    # Default extensions to include
    if include_extensions is None:
        include_extensions = {'.md', '.txt', '.facts', '.csv', '.json', '.html', '.htm'}
    else:
        # Ensure all extensions start with a dot
        include_extensions = {ext if ext.startswith('.') else f'.{ext}' for ext in include_extensions}

    # Walk through directory and collect files
    file_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in include_extensions:
                file_paths.append(os.path.join(root, file))

    # Limit files if specified
    if max_files is not None:
        random.shuffle(file_paths)
        file_paths = file_paths[:max_files]

    logger.info(f"Processing {len(file_paths)} files from {input_dir}")

    # Process each file
    for file_path in tqdm(file_paths, desc="Processing files"):
        # Load document
        doc = load_document(file_path)

        # Skip empty or error documents
        if not doc["content"] or "error" in doc:
            continue

        # Chunk document
        chunks = chunk_document(doc, max_chunk_size=max_chunk_size)
        all_chunks.extend(chunks)
        file_count += 1

    logger.info(f"Processed {file_count} files into {len(all_chunks)} chunks")
    return all_chunks


def parse_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse the response from OpenAI API.

    Args:
        response_text: JSON response from the OpenAI API

    Returns:
        List of examples
    """
    try:
        response_data = json.loads(response_text)
        examples = []

        # Extract examples from multiple places in the response

        # Look for direct Example and Example2, Example3, etc. keys
        for key, value in response_data.items():
            if key.startswith("Example") and isinstance(value, dict):
                # Check if the example has the right structure
                if all(field in value for field in ["query", "positive_document", "negative_documents"]):
                    examples.append(value)
                # If not, check if it has a properties field (nested structure)
                elif "properties" in value and isinstance(value["properties"], dict):
                    props = value["properties"]
                    if all(field in props for field in ["query", "positive_document", "negative_documents"]):
                        example = {
                            "query": props["query"],
                            "positive_document": props["positive_document"],
                            "negative_documents": props["negative_documents"]
                        }
                        examples.append(example)

        # Check in $defs section too
        if "$defs" in response_data and isinstance(response_data["$defs"], dict):
            for key, value in response_data["$defs"].items():
                if key.startswith("Example") and isinstance(value, dict):
                    # Direct format or properties format
                    if all(field in value for field in ["query", "positive_document", "negative_documents"]):
                        examples.append(value)
                    elif "properties" in value and isinstance(value["properties"], dict):
                        props = value["properties"]
                        if all(field in props for field in ["query", "positive_document", "negative_documents"]):
                            example = {
                                "query": props["query"],
                                "positive_document": props["positive_document"],
                                "negative_documents": props["negative_documents"]
                            }
                            examples.append(example)

        # Also try to look for examples array
        if "examples" in response_data and isinstance(response_data["examples"], list):
            examples.extend(response_data["examples"])

        return examples

    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        logger.error(f"Response text: {response_text}")
        return []


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_calls_per_minute: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_calls_per_minute: Maximum number of API calls per minute
        """
        self.max_calls_per_minute = max_calls_per_minute
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        """Wait if rate limit is reached."""
        async with self.lock:
            now = time.time()
            
            # Remove calls older than 1 minute
            self.calls = [t for t in self.calls if now - t < 60]
            
            # If at rate limit, wait until we can make another call
            if len(self.calls) >= self.max_calls_per_minute:
                oldest_call = min(self.calls)
                wait_time = 60 - (now - oldest_call)
                if wait_time > 0:
                    logger.debug(f"Rate limit reached. Waiting {wait_time:.2f} seconds.")
                    await asyncio.sleep(wait_time)
            
            # Add current time to calls
            self.calls.append(time.time())


async def generate_contrastive_negative(
    client: AsyncOpenAI,
    query: str,
    positive_document: Dict[str, Any],
    strategy_index: int,
    model: str = "gpt-4o-mini",
    retry_count: int = 3,
    retry_delay: float = 1.0,
    rate_limiter: Optional["RateLimiter"] = None
) -> Dict[str, Any]:
    """
    Generate a contrastive negative document for a query.
    
    Args:
        client: AsyncOpenAI client
        query: Query text
        positive_document: Positive document data
        strategy_index: Index of contrastive strategy to use
        model: Model to use for generation
        retry_count: Number of retries on error
        retry_delay: Delay between retries
        rate_limiter: Optional rate limiter
        
    Returns:
        Dictionary with negative document data
    """
    # Get contrastive strategy
    strategy = get_contrastive_strategy(strategy_index)
    strategy_name = strategy.split(":")[0].replace("Apply ", "").strip()
    
    # Prepare system prompt
    system_prompt = """You are an expert at creating training data for information retrieval systems.
Your task is to create a negative document example that looks relevant to the query but doesn't actually answer it.
This contrastive example should follow a specific strategy to make it challenging but still clearly non-relevant."""

    # Prepare user prompt
    user_prompt = f"""Generate a contrastive negative document for the following query and positive document:

QUERY: {query}

POSITIVE DOCUMENT:
{positive_document.get("document", "")}

{strategy}

Create a document that:
1. Uses similar terminology, style, and domain knowledge as the positive document
2. Appears relevant at first glance and would be retrieved by keyword matching
3. Deliberately fails to answer the specific question
4. Makes subtle but important distinctions that an information retrieval system should learn

Format your response as a JSON object with:
- "document": The text content of the negative document
- "explanation": An explanation of how this document follows the contrastive strategy
- "strategy": "{strategy_name}"
"""

    # Retry loop for API calls
    for attempt in range(retry_count):
        try:
            # Apply rate limiting if provided
            if rate_limiter:
                await rate_limiter.wait_if_needed()
                
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,  # Higher temperature for creativity
                max_tokens=2000,
                response_format={"type": "json_object"},
                seed=42 + attempt  # Different seed for each retry
            )

            # Extract and parse response
            response_text = response.choices[0].message.content
            
            try:
                data = json.loads(response_text)
                if "document" in data and "explanation" in data:
                    return data
                logger.warning(f"Invalid contrastive response format: {response_text[:100]}...")
            except json.JSONDecodeError:
                logger.warning("Failed to parse contrastive response as JSON")
                
            if attempt < retry_count - 1:
                await asyncio.sleep(retry_delay)
            
        except Exception as e:
            logger.warning(f"API error in contrastive generation: {e}")
            if attempt < retry_count - 1:
                await asyncio.sleep(retry_delay)
                
    # If all retries failed, return a minimal negative document
    return {
        "document": f"[This is a non-relevant document about {query.split()[0]} that uses the {strategy_name} strategy]",
        "explanation": f"Failed to generate contrastive negative with {strategy_name} strategy after {retry_count} attempts.",
        "strategy": strategy_name
    }


async def generate_examples_with_openai_async(
        chunk: Dict[str, Any],
        other_chunks: List[Dict[str, Any]],
        client: AsyncOpenAI,
        rate_limiter: Optional[RateLimiter] = None,
        model: str = "gpt-4o-mini",
        example_count: int = 2,
        negative_count: int = 2,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        domain_template: str = "generic",
        contrastive: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate training examples using OpenAI API asynchronously.

    Args:
        chunk: Document chunk to generate examples from
        other_chunks: Other document chunks for negative examples
        client: AsyncOpenAI client
        rate_limiter: Optional rate limiter for API calls
        model: OpenAI model to use
        example_count: Number of examples to generate per chunk
        negative_count: Number of negative examples per positive example
        temperature: Temperature for generation
        max_tokens: Maximum tokens for response
        retry_count: Number of retries on error
        retry_delay: Delay between retries
        domain_template: Domain template to use
        contrastive: Whether to use contrastive generation for negatives

    Returns:
        List of generated training examples
    """
    # Extract content and metadata
    content = chunk["content"]
    file_name = chunk["file_name"]

    # Select a subset of other chunks for negative examples
    # Limit to 3 chunks to keep prompt size reasonable
    selected_negative_chunks = []
    if other_chunks and not contrastive:
        # Only select negative chunks if not using contrastive generation
        # Select random chunks, but prefer chunks from different files
        other_file_chunks = [c for c in other_chunks if c["file_name"] != file_name]
        if other_file_chunks:
            selected_negative_chunks = random.sample(
                other_file_chunks,
                min(3, len(other_file_chunks))
            )
        elif other_chunks:
            selected_negative_chunks = random.sample(
                other_chunks,
                min(3, len(other_chunks))
            )

    # Create a string of potential negative examples
    negative_content = ""
    if not contrastive and selected_negative_chunks:
        for i, neg_chunk in enumerate(selected_negative_chunks):
            # Only include a snippet to keep the prompt size reasonable
            snippet = neg_chunk["content"][:500] + "..." if len(neg_chunk["content"]) > 500 else neg_chunk["content"]
            negative_content += f"\nPOTENTIAL NEGATIVE CONTENT {i + 1}:\n{snippet}\n"

    # Get domain template
    template = DOMAIN_TEMPLATES.get(domain_template, DOMAIN_TEMPLATES["generic"])
    
    # Prepare system prompt from template
    system_prompt = template["system_prompt"]

    # Prepare user prompt
    user_prompt = f"""Here is a document chunk to create training examples from:

DOCUMENT: {content}

{negative_content if negative_content else ""}

Create {example_count} training examples based on this content. Each should have:
1. A natural search query someone might ask
2. The positive document passage that answers the query
3. {negative_count} negative examples that don't actually answer the query

NOTE: For negative examples, use the provided potential negative content where relevant, 
or think about text that might be retrieved by a keyword search but doesn't actually provide the answer.

Format your response as a JSON object with Example1, Example2, etc. keys. Each example should have 'query', 'positive_document', and 'negative_documents' fields. Each negative_document should have 'document' and 'explanation' fields."""

    # Retry loop for API calls
    for attempt in range(retry_count):
        try:
            # Wait if rate limit is reached
            if rate_limiter:
                await rate_limiter.wait_if_needed()
            
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                seed=42 + attempt  # Different seed for each retry
            )

            # Extract and parse response
            response_text = response.choices[0].message.content
            logger.info(f"API response received, length: {len(response_text)}")

            # Parse response
            examples = parse_response(response_text)
            
            # If using contrastive generation, replace negative examples
            if contrastive and examples:
                # Process each example
                for example in examples:
                    # Get query and positive document
                    query = example.get("query", "")
                    positive_doc = {
                        "document": example.get("positive_document", "")
                    }
                    
                    # Generate contrastive negatives
                    contrastive_negatives = []
                    for i in range(negative_count):
                        # Apply rate limiting if provided
                        if rate_limiter:
                            await rate_limiter.wait_if_needed()
                        
                        negative = await generate_contrastive_negative(
                            client=client,
                            query=query,
                            positive_document=positive_doc,
                            strategy_index=i,
                            model=model,
                            retry_count=retry_count,
                            retry_delay=retry_delay,
                            rate_limiter=rate_limiter
                        )
                        contrastive_negatives.append(negative)
                    
                    # Replace negative documents with contrastive ones
                    example["negative_documents"] = contrastive_negatives

            if examples:
                logger.info(f"Successfully extracted {len(examples)} examples from response")
                return examples
            else:
                logger.warning("No examples found in response")
                logger.debug(f"Response: {response_text[:500]}...")
                if attempt < retry_count - 1:
                    await asyncio.sleep(retry_delay)
                continue

        except Exception as e:
            logger.warning(f"API error: {e}")
            if attempt < retry_count - 1:
                await asyncio.sleep(retry_delay)
            continue

    # Return empty list if all retries failed
    logger.error(f"Failed to generate examples for chunk {chunk.get('file_name')}")
    return []


# Function to generate examples using synchronous API (for backward compatibility)
def generate_examples_with_openai(
        chunk: Dict[str, Any],
        other_chunks: List[Dict[str, Any]],
        client: Any,  # OpenAI client
        model: str = "gpt-4o-mini",
        example_count: int = 2,
        negative_count: int = 2,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        domain_template: str = "generic"
) -> List[Dict[str, Any]]:
    """
    Generate training examples using OpenAI API (synchronous version).
    This function maintains backward compatibility with the original implementation.

    Args:
        chunk: Document chunk to generate examples from
        other_chunks: Other document chunks for negative examples
        client: OpenAI client
        model: OpenAI model to use
        example_count: Number of examples to generate per chunk
        negative_count: Number of negative examples per positive example
        temperature: Temperature for generation
        max_tokens: Maximum tokens for response
        retry_count: Number of retries on error
        retry_delay: Delay between retries
        domain_template: Domain template to use

    Returns:
        List of generated training examples
    """
    # Extract content and metadata
    content = chunk["content"]
    file_name = chunk["file_name"]

    # Select a subset of other chunks for negative examples
    # Limit to 3 chunks to keep prompt size reasonable
    selected_negative_chunks = []
    if other_chunks:
        # Select random chunks, but prefer chunks from different files
        other_file_chunks = [c for c in other_chunks if c["file_name"] != file_name]
        if other_file_chunks:
            selected_negative_chunks = random.sample(
                other_file_chunks,
                min(3, len(other_file_chunks))
            )
        elif other_chunks:
            selected_negative_chunks = random.sample(
                other_chunks,
                min(3, len(other_chunks))
            )

    # Create a string of potential negative examples
    negative_content = ""
    for i, neg_chunk in enumerate(selected_negative_chunks):
        # Only include a snippet to keep the prompt size reasonable
        snippet = neg_chunk["content"][:500] + "..." if len(neg_chunk["content"]) > 500 else neg_chunk["content"]
        negative_content += f"\nPOTENTIAL NEGATIVE CONTENT {i + 1}:\n{snippet}\n"

    # Get domain template
    template = DOMAIN_TEMPLATES.get(domain_template, DOMAIN_TEMPLATES["generic"])
    
    # Prepare system prompt from template
    system_prompt = template["system_prompt"]

    # Prepare user prompt
    user_prompt = f"""Here is a document chunk to create training examples from:

DOCUMENT: {content}

{negative_content if negative_content else ""}

Create {example_count} training examples based on this content. Each should have:
1. A natural search query someone might ask
2. The positive document passage that answers the query
3. {negative_count} negative examples that don't actually answer the query

NOTE: For negative examples, use the provided potential negative content where relevant, 
or think about text that might be retrieved by a keyword search but doesn't actually provide the answer.

Format your response as a JSON object with Example1, Example2, etc. keys. Each example should have 'query', 'positive_document', and 'negative_documents' fields. Each negative_document should have 'document' and 'explanation' fields."""

    # Retry loop for API calls
    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                seed=42 + attempt  # Different seed for each retry
            )

            # Extract and parse response
            response_text = response.choices[0].message.content
            logger.info(f"API response received, length: {len(response_text)}")

            # Parse response
            examples = parse_response(response_text)

            if examples:
                logger.info(f"Successfully extracted {len(examples)} examples from response")
                return examples
            else:
                logger.warning("No examples found in response")
                logger.debug(f"Response: {response_text[:500]}...")
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)
                continue

        except Exception as e:
            logger.warning(f"API error: {e}")
            if attempt < retry_count - 1:
                time.sleep(retry_delay)
            continue

    # Return empty list if all retries failed
    logger.error(f"Failed to generate examples for chunk {chunk.get('file_name')}")
    return []


class WorkerQueue:
    """Worker queue for processing chunks in parallel."""
    
    def __init__(
        self, 
        client: AsyncOpenAI,
        rate_limiter: RateLimiter,
        model: str,
        example_count: int,
        negative_count: int,
        temperature: float,
        max_tokens: int,
        retry_count: int,
        retry_delay: float,
        domain_template: str,
        contrastive: bool,
        num_workers: int = 4
    ):
        """
        Initialize worker queue.
        
        Args:
            client: AsyncOpenAI client
            rate_limiter: Rate limiter for API calls
            model: OpenAI model to use
            example_count: Number of examples to generate per chunk
            negative_count: Number of negative examples per positive example
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            retry_count: Number of retries on error
            retry_delay: Delay between retries
            domain_template: Domain template to use
            contrastive: Whether to use contrastive generation
            num_workers: Number of parallel workers
        """
        self.client = client
        self.rate_limiter = rate_limiter
        self.model = model
        self.example_count = example_count
        self.negative_count = negative_count
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.domain_template = domain_template
        self.contrastive = contrastive
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(num_workers)
        self.progress = tqdm(desc="Generating examples")
        self.total_examples = 0
    
    async def process_chunk(
        self, 
        chunk: Dict[str, Any], 
        other_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process a single chunk using a worker.
        
        Args:
            chunk: Document chunk to process
            other_chunks: Other chunks for negative examples
            
        Returns:
            List of generated examples
        """
        async with self.semaphore:
            result = await generate_examples_with_openai_async(
                chunk=chunk,
                other_chunks=other_chunks,
                client=self.client,
                rate_limiter=self.rate_limiter,
                model=self.model,
                example_count=self.example_count,
                negative_count=self.negative_count,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                retry_count=self.retry_count,
                retry_delay=self.retry_delay,
                domain_template=self.domain_template,
                contrastive=self.contrastive
            )
            
            # Update progress
            self.progress.update(1)
            self.total_examples += len(result)
            self.progress.set_postfix(examples=self.total_examples)
            
            return result
    
    async def process_chunks(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process all chunks in parallel.
        
        Args:
            chunks: List of document chunks to process
            
        Returns:
            List of all generated examples
        """
        # Create tasks
        tasks = []
        for chunk in chunks:
            # Use all other chunks as potential negative examples
            other_chunks = [c for c in chunks if c != chunk]
            task = self.process_chunk(chunk, other_chunks)
            tasks.append(task)
        
        # Set total for progress bar
        self.progress.reset(total=len(chunks))
        
        # Run tasks and gather results
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_examples = []
        for examples in results:
            all_examples.extend(examples)
        
        # Close progress bar
        self.progress.close()
        
        return all_examples


async def process_chunks_async(
        chunks: List[Dict[str, Any]],
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        example_count: int = 2,
        negative_count: int = 2,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        domain_template: str = "generic",
        contrastive: bool = False,
        num_workers: int = 4,
        rate_limit: int = 60
) -> List[Dict[str, Any]]:
    """
    Process chunks in parallel using async workers.

    Args:
        chunks: List of document chunks to process
        api_key: OpenAI API key
        api_base: Base URL for OpenAI API
        model: OpenAI model to use
        example_count: Number of examples to generate per chunk
        negative_count: Number of negative examples per positive example
        temperature: Temperature for generation
        max_tokens: Maximum tokens for response
        retry_count: Number of retries on error
        retry_delay: Delay between retries
        domain_template: Domain template to use
        contrastive: Whether to use contrastive generation
        num_workers: Number of parallel workers
        rate_limit: Maximum number of API calls per minute

    Returns:
        List of generated examples
    """
    if AsyncOpenAI is None:
        logger.error("AsyncOpenAI client not available. Install the openai package.")
        return []
        
    # Create AsyncOpenAI client
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    
    # Create rate limiter
    rate_limiter = RateLimiter(max_calls_per_minute=rate_limit)
    
    # Create worker queue
    worker_queue = WorkerQueue(
        client=client,
        rate_limiter=rate_limiter,
        model=model,
        example_count=example_count,
        negative_count=negative_count,
        temperature=temperature,
        max_tokens=max_tokens,
        retry_count=retry_count,
        retry_delay=retry_delay,
        domain_template=domain_template,
        contrastive=contrastive,
        num_workers=num_workers
    )
    
    # Process chunks
    logger.info(f"Processing {len(chunks)} chunks with {num_workers} workers")
    logger.info(f"Using domain template: {domain_template}")
    if contrastive:
        logger.info("Using contrastive generation for negative examples")
    
    examples = await worker_queue.process_chunks(chunks)
    
    logger.info(f"Generated {len(examples)} examples from {len(chunks)} chunks")
    return examples


# Original synchronous function (for backward compatibility)
def process_chunks(
        chunks: List[Dict[str, Any]],
        api_key: str,
        model: str = "gpt-4o-mini",
        example_count: int = 2,
        negative_count: int = 2,
        batch_size: int = 5,
        temperature: float = 0.7,
        domain_template: str = "generic",
        seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Process chunks in batches (synchronous version for backward compatibility).

    Args:
        chunks: List of document chunks
        api_key: OpenAI API key
        model: OpenAI model to use
        example_count: Number of examples to generate per chunk
        negative_count: Number of negative examples per positive example
        batch_size: Number of chunks to process in each batch
        temperature: Temperature for generation
        domain_template: Domain template to use
        seed: Random seed for reproducibility

    Returns:
        List of generated training examples
    """
    if OpenAI is None:
        logger.error("OpenAI client not available. Install the openai package.")
        return []
        
    # Create OpenAI client
    client = OpenAI(api_key=api_key)

    # Set random seed
    random.seed(seed)

    # Setup progress tracking
    all_examples = []

    # Process chunks in batches
    for i in tqdm(range(0, len(chunks), batch_size), desc="Processing chunks"):
        batch = chunks[i:i + batch_size]
        other_chunks = [c for c in chunks if c not in batch]

        # Process each chunk in the batch
        for chunk in batch:
            examples = generate_examples_with_openai(
                chunk,
                other_chunks,
                client,
                model,
                example_count,
                negative_count,
                temperature,
                domain_template=domain_template
            )
            all_examples.extend(examples)

    logger.info(f"Generated {len(all_examples)} examples from {len(chunks)} chunks")
    return all_examples


def save_to_file(data: List[Dict[str, Any]], output_file: str, output_format: str = "splade"):
    """
    Save training data to file in specified format.

    Args:
        data: List of training examples
        output_file: Path to output file
        output_format: Output format (splade, json, jsonl, csv, tsv)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get appropriate file extension for format
    formats = {
        "splade": ".json",
        "json": ".json",
        "jsonl": ".jsonl",
        "csv": ".csv",
        "tsv": ".tsv"
    }
    
    # Update extension if needed
    base_path = Path(output_file)
    if output_format in formats and not output_file.endswith(formats[output_format]):
        output_file = str(base_path.with_suffix(formats[output_format]))
    
    if output_format == "splade":
        # Convert to format expected by SPLADE training script
        formatted_data = []
        for example in data:
            # Check for format inconsistencies
            if "negative_documents" not in example:
                logger.warning(f"Skipping example missing negative_documents: {example}")
                continue

            # Handle different formats of negative_documents
            negative_docs = []
            explanations = []
            strategies = []

            for neg_doc in example["negative_documents"]:
                if isinstance(neg_doc, dict) and "document" in neg_doc and "explanation" in neg_doc:
                    negative_docs.append(neg_doc["document"])
                    explanations.append(neg_doc["explanation"])
                    if "strategy" in neg_doc:
                        strategies.append(neg_doc["strategy"])
                elif isinstance(neg_doc, str):
                    # Handle case where negative_documents is just a list of strings
                    negative_docs.append(neg_doc)
                    explanations.append("No explanation provided")
                    strategies.append("")

            formatted_example = {
                "query": example["query"],
                "positive_document": example["positive_document"],
                "negative_documents": negative_docs,
                "explanations": explanations
            }
            
            # Add strategies if present
            if any(strategies) and all(strategies):
                formatted_example["strategies"] = strategies
                
            formatted_data.append(formatted_example)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            
    elif output_format == "json":
        # Save as raw JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    elif output_format == "jsonl":
        # Save as JSONL (one JSON object per line)
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
                
    elif output_format in ["csv", "tsv"]:
        # Save as CSV or TSV
        delimiter = ',' if output_format == "csv" else '\t'
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            
            # Write header
            writer.writerow(["query", "document", "is_relevant", "explanation"])
            
            # Write data
            for example in data:
                query = example.get("query", "")
                positive_document = example.get("positive_document", "")
                
                # Write positive example
                writer.writerow([query, positive_document, 1, "Positive document"])
                
                # Write negative examples
                for neg_doc in example.get("negative_documents", []):
                    if isinstance(neg_doc, dict) and "document" in neg_doc:
                        explanation = neg_doc.get("explanation", "")
                        writer.writerow([query, neg_doc["document"], 0, explanation])
                    elif isinstance(neg_doc, str):
                        writer.writerow([query, neg_doc, 0, ""])

    logger.info(f"Saved {len(data)} training examples to {output_file} in {output_format} format")


def split_train_val_test(data: List[Dict[str, Any]], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into training, validation, and test sets.

    Args:
        data: List of training examples
        train_ratio: Ratio of training examples
        val_ratio: Ratio of validation examples

    Returns:
        Tuple of (training_data, validation_data, test_data)
    """
    # Shuffle data
    data_copy = data.copy()
    random.shuffle(data_copy)

    # Calculate sizes
    train_size = int(len(data_copy) * train_ratio)
    val_size = int(len(data_copy) * val_ratio)
    
    # Split into train, validation, and test sets
    train_data = data_copy[:train_size]
    val_data = data_copy[train_size:train_size + val_size]
    test_data = data_copy[train_size + val_size:]

    logger.info(f"Split data into {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test examples")
    return train_data, val_data, test_data


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate training data for SPLADE model fine-tuning")

    parser.add_argument('--input-dir', '-i', required=True,
                        help='Directory containing document files')

    parser.add_argument('--output-file', '-o', required=True,
                        help='Output file for training data')

    parser.add_argument('--output-format', choices=['splade', 'json', 'jsonl', 'csv', 'tsv'], default='splade',
                        help='Output format (default: splade)')

    parser.add_argument('--validation-file',
                        help='Output file for validation data. If not specified, no separate validation file is created.')

    parser.add_argument('--test-file',
                        help='Output file for test data. If not specified, no separate test file is created.')

    parser.add_argument('--split', action='store_true',
                        help='Split data into train, validation, and test sets')

    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Ratio of training examples when splitting (default: 0.8)')

    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Ratio of validation examples when splitting (default: 0.1)')

    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process (default: all)')

    parser.add_argument('--max-chunk-size', type=int, default=2000,
                        help='Maximum chunk size in characters (default: 2000)')

    parser.add_argument('--example-count', type=int, default=2,
                        help='Number of examples to generate per chunk (default: 2)')

    parser.add_argument('--negative-count', type=int, default=2,
                        help='Number of negative examples per positive example (default: 2)')

    parser.add_argument('--batch-size', type=int, default=5,
                        help='Number of chunks to process in each batch in synchronous mode (default: 5)')

    parser.add_argument('--model', default="gpt-4o-mini",
                        help='OpenAI model to use (default: gpt-4o-mini)')

    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for generation (default: 0.7)')

    parser.add_argument('--api-key',
                        help='OpenAI API key (default: from OPENAI_API_KEY environment variable)')

    parser.add_argument('--api-base', default="https://api.openai.com/v1",
                        help='Base URL for OpenAI-compatible API (default: https://api.openai.com/v1)')

    parser.add_argument('--domain-template', choices=list(DOMAIN_TEMPLATES.keys()), default='generic',
                        help='Domain template to use (default: generic)')

    parser.add_argument('--extensions', nargs='+', default=None,
                        help='File extensions to include (default: md, txt, facts, csv, json, html, htm)')

    parser.add_argument('--contrastive', action='store_true',
                        help='Use contrastive generation for negative examples')

    parser.add_argument('--async', dest='use_async', action='store_true',
                        help='Use asynchronous processing with workers (recommended)')

    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers for async mode (default: 4)')

    parser.add_argument('--rate-limit', type=int, default=60,
                        help='Maximum API calls per minute for async mode (default: 60)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    return parser.parse_args()


async def main_async(args):
    """Main async function to generate training data."""
    # Set random seed for reproducibility
    random.seed(args.seed)

    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or use --api-key")
        return

    # Process input directory
    include_extensions = set(args.extensions) if args.extensions else None
    logger.info(f"Processing directory: {args.input_dir}")
    chunks = process_directory(
        args.input_dir,
        args.max_files,
        args.max_chunk_size,
        include_extensions
    )

    if not chunks:
        logger.error("No valid document chunks found. Check input directory.")
        return

    # Generate training examples
    logger.info(f"Generating training examples using {args.model} with {args.workers} workers (async mode)")
    examples = await process_chunks_async(
        chunks,
        api_key,
        args.api_base,
        args.model,
        args.example_count,
        args.negative_count,
        args.temperature,
        max_tokens=2000,
        retry_count=3,
        retry_delay=1.0,
        domain_template=args.domain_template,
        contrastive=args.contrastive,
        num_workers=args.workers,
        rate_limit=args.rate_limit
    )

    if not examples:
        logger.error("Failed to generate training examples")
        return

    # Split data if needed
    if args.split or (args.validation_file and args.test_file):
        train_data, val_data, test_data = split_train_val_test(
            examples, 
            args.train_ratio, 
            args.val_ratio
        )
        
        # Save training data
        save_to_file(train_data, args.output_file, args.output_format)
        
        # Save validation data if requested
        if args.validation_file:
            save_to_file(val_data, args.validation_file, args.output_format)
        
        # Save test data if requested
        if args.test_file:
            save_to_file(test_data, args.test_file, args.output_format)
    elif args.validation_file:
        # Split into train and validation only
        train_data, val_data = examples[int(len(examples) * args.val_ratio):], examples[:int(len(examples) * args.val_ratio)]
        
        # Save training data
        save_to_file(train_data, args.output_file, args.output_format)
        
        # Save validation data
        save_to_file(val_data, args.validation_file, args.output_format)
    else:
        # Save all data as training
        save_to_file(examples, args.output_file, args.output_format)

    logger.info("Training data generation completed")


def main():
    """Main function to generate training data."""
    args = parse_arguments()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Check if async mode is requested
    if args.use_async:
        if AsyncOpenAI is None:
            logger.error("AsyncOpenAI not available. Install openai package for async mode.")
            return
        asyncio.run(main_async(args))
        return

    # Synchronous mode (original behavior)
    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or use --api-key")
        return

    # Process input directory
    include_extensions = set(args.extensions) if args.extensions else None
    logger.info(f"Processing directory: {args.input_dir}")
    chunks = process_directory(
        args.input_dir,
        args.max_files,
        args.max_chunk_size,
        include_extensions
    )

    if not chunks:
        logger.error("No valid document chunks found. Check input directory.")
        return

    # Generate training examples
    logger.info(f"Generating training examples using {args.model} (synchronous mode)")
    examples = process_chunks(
        chunks,
        api_key,
        args.model,
        args.example_count,
        args.negative_count,
        args.batch_size,
        args.temperature,
        args.domain_template,
        args.seed
    )

    if not examples:
        logger.error("Failed to generate training examples")
        return

    # Split data as needed
    if args.split or (args.validation_file and args.test_file):
        train_data, val_data, test_data = split_train_val_test(examples, args.train_ratio, args.val_ratio)
        
        # Save training data
        save_to_file(train_data, args.output_file, args.output_format)
        
        # Save validation data if requested
        if args.validation_file:
            save_to_file(val_data, args.validation_file, args.output_format)
        
        # Save test data if requested
        if args.test_file:
            save_to_file(test_data, args.test_file, args.output_format)
    elif args.validation_file:
        # Basic train/val split
        val_size = int(len(examples) * args.val_ratio)
        train_data = examples[val_size:]
        val_data = examples[:val_size]

        # Save training data
        save_to_file(train_data, args.output_file, args.output_format)

        # Save validation data
        save_to_file(val_data, args.validation_file, args.output_format)
    else:
        # Save all data as training
        save_to_file(examples, args.output_file, args.output_format)

    logger.info("Training data generation completed")


if __name__ == "__main__":
    main()
