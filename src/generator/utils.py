"""
Utility functions for SPLADE training data generation.

This module provides helper functions for various aspects of training data generation.
"""

import os
import re
import json
import random
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logger = logging.getLogger("generator.utils")


class TensorJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles PyTorch Tensors.
    
    This encoder converts Tensors to Python lists or scalar values
    that can be safely serialized to JSON.
    """
    def default(self, obj):
        import torch
        if isinstance(obj, torch.Tensor):
            # Handle different tensor shapes
            if obj.numel() == 1:  # Single value tensor
                return obj.item()  # Convert to Python scalar
            else:
                return obj.tolist()  # Convert to Python list
        # Let the parent class handle other types
        return super().default(obj)


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
    """
    Get a contrastive strategy by index.
    
    Args:
        index: Strategy index
        
    Returns:
        Strategy description
    """
    return CONTRASTIVE_STRATEGIES[index % len(CONTRASTIVE_STRATEGIES)]


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


def split_train_val_test(
    data: List[Dict[str, Any]], 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
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
    base_path = os.path.splitext(output_file)[0]
    if output_format in formats and not output_file.endswith(formats[output_format]):
        output_file = base_path + formats[output_format]
    
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
        import csv
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
