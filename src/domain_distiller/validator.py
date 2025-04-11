"""
Dataset validator for checking the quality of generated training data.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple, Set

from .utils.api_utils import APIClient, AsyncWorkerPool
from .templates import get_domain_template
from .languages import get_language_config

# Configure logger
logger = logging.getLogger(__name__)


def load_dataset(dataset_file: str) -> List[Dict[str, Any]]:
    """
    Load dataset from file.
    
    Args:
        dataset_file: Path to dataset file
        
    Returns:
        List of dataset examples
    """
    try:
        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        logger.info(f"Loaded {len(dataset)} examples from {dataset_file}")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise ValueError(f"Failed to load dataset from {dataset_file}: {str(e)}")


async def validate_example(
    api_client: APIClient,
    example: Dict[str, Any],
    domain: str,
    language: str,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Validate a single dataset example.
    
    Args:
        api_client: API client for validation
        example: Dataset example
        domain: Domain name
        language: Language code
        strict: Whether to perform strict validation
        
    Returns:
        Dictionary with validation results
    """
    query = example.get("query", "")
    positive_document = example.get("positive_document", "")
    negative_documents = example.get("negative_documents", [])
    
    # Extract negative document texts
    negative_texts = []
    for neg_doc in negative_documents:
        if isinstance(neg_doc, dict) and "document" in neg_doc:
            negative_texts.append(neg_doc["document"])
        elif isinstance(neg_doc, str):
            negative_texts.append(neg_doc)
    
    # Prepare system prompt
    system_prompt = f"""You are an expert evaluator of training data for information retrieval systems.
Your task is to assess the quality of query-document pairs for the {domain} domain.
Evaluate whether the positive document actually answers the query and whether the negative documents do not.
Your responses should be objective and critical, identifying any issues with the data."""
    
    # Prepare user prompt
    user_prompt = f"""Evaluate the quality of this training example for the {domain} domain in {language} language:

QUERY: {query}

POSITIVE DOCUMENT:
{positive_document}

NEGATIVE DOCUMENTS:
{"".join([f"--- Negative Document {i+1} ---\n{text}\n\n" for i, text in enumerate(negative_texts)])}

Your evaluation should assess:
1. Query quality: Is it clear, well-formed, and realistic?
2. Positive document relevance: Does it fully answer the query?
3. Negative documents quality: Are they good distractors that don't actually answer the query?
4. Overall example quality: Is this a good training example?

For each negative document, explicitly state whether it truly does NOT answer the query.
Identify any cases where a negative document actually does answer the query (this would be an issue).

{"Apply strict criteria for relevance judgments." if strict else "Apply reasonable criteria for relevance judgments."}

Format your response as a JSON object with these fields:
- "query_quality": Rating from 1-5
- "query_issues": Array of identified issues with the query
- "positive_relevance": Rating from 1-5
- "positive_issues": Array of identified issues with the positive document
- "negative_evaluations": Array of objects, one for each negative document, each with:
  * "is_truly_negative": Boolean (true if it doesn't answer the query)
  * "issues": Array of identified issues
- "overall_quality": Rating from 1-5
- "recommendations": Array of suggestions to improve this example
"""

    # Schema for the response
    schema = {
        "type": "object",
        "properties": {
            "query_quality": {"type": "number"},
            "query_issues": {"type": "array", "items": {"type": "string"}},
            "positive_relevance": {"type": "number"},
            "positive_issues": {"type": "array", "items": {"type": "string"}},
            "negative_evaluations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "is_truly_negative": {"type": "boolean"},
                        "issues": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "overall_quality": {"type": "number"},
            "recommendations": {"type": "array", "items": {"type": "string"}}
        }
    }
    
    # Generate validation
    try:
        response = await api_client.structured_generation(
            prompt=user_prompt,
            system_prompt=system_prompt,
            schema=schema,
            temperature=0.3,  # Low temperature for consistent evaluation
            max_tokens=3000
        )
        
        # Add metadata
        response["_meta"] = {
            "query": query[:100] + "..." if len(query) > 100 else query,
            "domain": domain,
            "language": language,
            "strict": strict,
            "validated": True
        }
        
        # Add pass/fail flags
        response["passes_validation"] = (
            response.get("query_quality", 0) >= 3 and
            response.get("positive_relevance", 0) >= 3 and
            response.get("overall_quality", 0) >= 3 and
            all(item.get("is_truly_negative", False) for item in response.get("negative_evaluations", []))
        )
        
        logger.info(f"Validated example: {query[:50]}... - {'PASS' if response['passes_validation'] else 'FAIL'}")
        return response
        
    except Exception as e:
        logger.error(f"Error validating example: {str(e)}")
        return {
            "error": str(e),
            "_meta": {
                "query": query[:100] + "..." if len(query) > 100 else query,
                "domain": domain,
                "language": language,
                "strict": strict,
                "error": True
            },
            "passes_validation": False
        }


async def _process_example(
    example: Dict[str, Any],
    example_id: int,
    domain: str,
    language: str,
    strict: bool,
    api_client: APIClient
) -> Dict[str, Any]:
    """Process a single example for validation."""
    try:
        # Validate example
        result = await validate_example(
            api_client=api_client,
            example=example,
            domain=domain,
            language=language,
            strict=strict
        )
        
        # Add example ID to metadata
        if "_meta" in result:
            result["_meta"]["example_id"] = example_id
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing example {example_id}: {str(e)}")
        query = example.get("query", "unknown")
        return {
            "error": str(e),
            "_meta": {
                "query": query[:100] + "..." if len(query) > 100 else query,
                "domain": domain,
                "language": language,
                "strict": strict,
                "example_id": example_id,
                "error": True
            },
            "passes_validation": False
        }


async def _validate_dataset_async(
    dataset_file: str,
    strict: bool,
    api_base: str,
    api_key: str,
    model: str,
    num_workers: int
) -> Dict[str, Any]:
    """Asynchronous implementation of validate_dataset."""
    # Load dataset
    dataset = load_dataset(dataset_file)
    
    # Extract domain and language from first example if available
    domain = "unknown"
    language = "en"
    if dataset and len(dataset) > 0:
        first_example = dataset[0]
        if "_meta" in first_example:
            domain = first_example["_meta"].get("domain", domain)
            language = first_example["_meta"].get("language", language)
    
    # Initialize API client
    api_client = APIClient(
        api_base=api_base,
        api_key=api_key,
        model=model
    )
    
    # Process examples
    logger.info(f"Validating {len(dataset)} examples{'(strict mode)' if strict else ''}")
    
    # Create processing tasks
    tasks = []
    for i, example in enumerate(dataset):
        task = _process_example(
            example=example,
            example_id=i,
            domain=domain,
            language=language,
            strict=strict,
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
                example_idx = i + j
                if example_idx < len(dataset):
                    example = dataset[example_idx]
                    query = example.get("query", "unknown")
                    error_result = {
                        "error": str(result),
                        "_meta": {
                            "query": query[:100] + "..." if len(query) > 100 else query,
                            "domain": domain,
                            "language": language,
                            "strict": strict,
                            "example_id": example_idx,
                            "error": True
                        },
                        "passes_validation": False
                    }
                    results.append(error_result)
            else:
                results.append(result)
    
    # Calculate statistics
    total = len(results)
    passed = sum(1 for r in results if r.get("passes_validation", False))
    failed = total - passed
    errors = sum(1 for r in results if "_meta" in r and r["_meta"].get("error", False))
    
    # Create summary
    summary = {
        "dataset_file": dataset_file,
        "total_examples": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": passed / total if total > 0 else 0,
        "strict_mode": strict,
        "domain": domain,
        "language": language
    }
    
    # Save validation results
    output_dir = os.path.dirname(dataset_file)
    base_name = os.path.basename(dataset_file)
    name_parts = os.path.splitext(base_name)
    validation_file = os.path.join(output_dir, f"{name_parts[0]}_validation{name_parts[1]}")
    
    with open(validation_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Validation summary: {passed}/{total} examples passed ({summary['pass_rate']:.1%})")
    logger.info(f"Saved validation results to {validation_file}")
    
    # Close API client
    await api_client.close()
    
    return summary


def validate_dataset(
    dataset_file: str,
    strict: bool = False,
    api_base: str = "https://api.openai.com/v1",
    api_key: str = None,
    model: str = "gpt-4o",
    num_workers: int = 4
) -> Dict[str, Any]:
    """
    Validate the quality of a generated dataset.
    
    Args:
        dataset_file: Path to dataset file
        strict: Whether to perform strict validation
        api_base: Base URL for OpenAI-compatible API
        api_key: API key
        model: Model name
        num_workers: Number of async workers
        
    Returns:
        Dictionary with validation summary
    """
    # Create event loop and run async implementation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            _validate_dataset_async(
                dataset_file=dataset_file,
                strict=strict,
                api_base=api_base,
                api_key=api_key,
                model=model,
                num_workers=num_workers
            )
        )
        return result
    finally:
        loop.close()
