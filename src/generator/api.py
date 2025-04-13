"""
API client for SPLADE training data generation.

This module provides functions for interacting with OpenAI-compatible APIs for generating
training data examples.
"""

import logging
import asyncio
import time
import json
import random
from typing import List, Dict, Any, Optional

from src.generator.utils import parse_response, get_contrastive_strategy, RateLimiter
from src.generator.templates import get_template, get_language_name

# Configure logging
logger = logging.getLogger("generator.api")

# Conditional import for API client
try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    AsyncOpenAI = None
    OpenAI = None
    logger.warning("OpenAI library not found. Install with 'pip install openai' to use API features.")


async def generate_contrastive_negative(
    client: Any,
    query: str,
    positive_document: Dict[str, Any],
    strategy_index: int,
    model: str = "gpt-4o-mini",
    retry_count: int = 3,
    retry_delay: float = 1.0,
    language: Optional[str] = None,
    rate_limiter: Optional[RateLimiter] = None
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
        language: Optional language code for generation
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

    # Add language instruction if specified
    if language:
        language_name = get_language_name(language)
        system_prompt += f"\n\nIMPORTANT: Create your response in {language_name} language ONLY."

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
            except Exception as e:
                logger.warning(f"Failed to parse contrastive response as JSON: {e}")
                
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
    client: Any,
    rate_limiter: Optional[RateLimiter] = None,
    model: str = "gpt-4o-mini",
    example_count: int = 2,
    negative_count: int = 2,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    retry_count: int = 3,
    retry_delay: float = 1.0,
    domain_template: str = "generic",
    language: Optional[str] = None,
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
        language: Optional language code for generation
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

    # Get appropriate template for domain and language
    template = get_template(domain_template, language)
    
    # Prepare system prompt from template
    system_prompt = template["system_prompt"]
    
    # Log the template being used
    if language:
        logger.info(f"Using template: {template['name']} with language: {language}")
    else:
        logger.info(f"Using template: {template['name']}")

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

    # Add language instruction if specified and not already in template
    if language and "auto" not in template.get("language", ""):
        language_name = get_language_name(language)
        user_prompt += f"\n\nIMPORTANT: Create all examples in {language_name} language ONLY."

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
                            language=language,
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


def generate_examples_with_openai(
    chunk: Dict[str, Any],
    other_chunks: List[Dict[str, Any]],
    client: Any,
    model: str = "gpt-4o-mini",
    example_count: int = 2,
    negative_count: int = 2,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    retry_count: int = 3,
    retry_delay: float = 1.0,
    domain_template: str = "generic",
    language: Optional[str] = None
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
        language: Optional language code for generation

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

    # Get appropriate template for domain and language
    template = get_template(domain_template, language)
    
    # Prepare system prompt from template
    system_prompt = template["system_prompt"]
    
    # Log the template being used
    if language:
        logger.info(f"Using template: {template['name']} with language: {language}")
    else:
        logger.info(f"Using template: {template['name']}")

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

    # Add language instruction if specified and not already in template
    if language and "auto" not in template.get("language", ""):
        language_name = get_language_name(language)
        user_prompt += f"\n\nIMPORTANT: Create all examples in {language_name} language ONLY."

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
        client: Any,
        rate_limiter: RateLimiter,
        model: str,
        example_count: int,
        negative_count: int,
        temperature: float,
        max_tokens: int,
        retry_count: int,
        retry_delay: float,
        domain_template: str,
        language: Optional[str] = None,
        contrastive: bool = False,
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
            language: Optional language code for generation
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
        self.language = language
        self.contrastive = contrastive
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(num_workers)
        from tqdm import tqdm
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
                language=self.language,
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
    language: Optional[str] = None,
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
        language: Optional language code for generation
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
        language=language,
        contrastive=contrastive,
        num_workers=num_workers
    )
    
    # Process chunks
    logger.info(f"Processing {len(chunks)} chunks with {num_workers} workers")
    logger.info(f"Using domain template: {domain_template}")
    if language:
        logger.info(f"Using language: {language}")
    if contrastive:
        logger.info("Using contrastive generation for negative examples")
    
    examples = await worker_queue.process_chunks(chunks)
    
    logger.info(f"Generated {len(examples)} examples from {len(chunks)} chunks")
    return examples


def process_chunks(
    chunks: List[Dict[str, Any]],
    api_key: str,
    model: str = "gpt-4o-mini",
    example_count: int = 2,
    negative_count: int = 2,
    batch_size: int = 5,
    temperature: float = 0.7,
    domain_template: str = "generic",
    language: Optional[str] = None,
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
        language: Optional language code for generation
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
    from tqdm import tqdm

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
                max_tokens=max_tokens,
                retry_count=retry_count,
                retry_delay=retry_delay,
                domain_template=domain_template,
                language=language
            )
            all_examples.extend(examples)

    logger.info(f"Generated {len(all_examples)} examples from {len(chunks)} chunks")
    return all_examples
