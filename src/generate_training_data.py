#!/usr/bin/env python3
"""
Final solution for the SPLADE training data generator.
This version correctly parses the OpenAI API response format.
"""

import argparse
import json
import logging
import os
import random
import re
import time
from textwrap import dedent
from typing import List, Dict, Any, Optional

from openai import OpenAI
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
        if extension == '.md' or extension == '.txt':
            # For markdown and text files, keep content as is
            pass
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


def process_directory(input_dir: str, max_files: Optional[int] = None, max_chunk_size: int = 2000) -> List[
    Dict[str, Any]]:
    """
    Process all documents in a directory and its subdirectories.

    Args:
        input_dir: Directory containing documents
        max_files: Maximum number of files to process (None for all)
        max_chunk_size: Maximum chunk size in characters

    Returns:
        List of processed document chunks
    """
    all_chunks = []
    file_count = 0

    # Walk through directory and collect files
    file_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.md', '.txt', '.facts')):
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


def generate_examples_with_openai(
        chunk: Dict[str, Any],
        other_chunks: List[Dict[str, Any]],
        client: OpenAI,
        model: str = "gpt-4o-mini",
        example_count: int = 2,
        negative_count: int = 2,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        retry_count: int = 3,
        retry_delay: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Generate training examples using OpenAI API.

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

    # Prepare system prompt
    system_prompt = dedent(f"""
    You are an expert at creating training data for information retrieval models. 
    Your task is to create {example_count} realistic search queries that someone might use to find specific information in the provided document.

    For each query:
    1. Create a natural, specific question someone might search for
    2. Identify the exact text passage that answers this query
    3. Find {negative_count} negative examples - text that looks similar but doesn't answer the query

    The document content is technical documentation about heating systems, heat pumps and related topics.
    """)

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


def process_chunks(
        chunks: List[Dict[str, Any]],
        api_key: str,
        model: str = "gpt-4o-mini",
        example_count: int = 2,
        negative_count: int = 2,
        batch_size: int = 5,
        temperature: float = 0.7,
        seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Process chunks in batches.

    Args:
        chunks: List of document chunks
        api_key: OpenAI API key
        model: OpenAI model to use
        example_count: Number of examples to generate per chunk
        negative_count: Number of negative examples per positive example
        batch_size: Number of chunks to process in each batch
        temperature: Temperature for generation
        seed: Random seed for reproducibility

    Returns:
        List of generated training examples
    """
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
                temperature
            )
            all_examples.extend(examples)

    logger.info(f"Generated {len(all_examples)} examples from {len(chunks)} chunks")
    return all_examples


def save_to_file(data: List[Dict[str, Any]], output_file: str):
    """
    Save training data to file in JSON format.

    Args:
        data: List of training examples
        output_file: Path to output file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

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

        for neg_doc in example["negative_documents"]:
            if isinstance(neg_doc, dict) and "document" in neg_doc and "explanation" in neg_doc:
                negative_docs.append(neg_doc["document"])
                explanations.append(neg_doc["explanation"])
            elif isinstance(neg_doc, str):
                # Handle case where negative_documents is just a list of strings
                negative_docs.append(neg_doc)
                explanations.append("No explanation provided")

        formatted_example = {
            "query": example["query"],
            "positive_document": example["positive_document"],
            "negative_documents": negative_docs,
            "explanations": explanations
        }
        formatted_data.append(formatted_example)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(formatted_data)} training examples to {output_file}")


def split_train_val(data: List[Dict[str, Any]], val_ratio: float = 0.1) -> tuple:
    """
    Split data into training and validation sets.

    Args:
        data: List of training examples
        val_ratio: Ratio of validation examples

    Returns:
        Tuple of (training_data, validation_data)
    """
    # Shuffle data
    data_copy = data.copy()
    random.shuffle(data_copy)

    # Split into train and validation sets
    val_size = int(len(data_copy) * val_ratio)
    train_data = data_copy[val_size:]
    val_data = data_copy[:val_size]

    logger.info(f"Split data into {len(train_data)} training and {len(val_data)} validation examples")
    return train_data, val_data


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate training data for SPLADE model fine-tuning")

    parser.add_argument('--input-dir', '-i', required=True,
                        help='Directory containing document files')

    parser.add_argument('--output-file', '-o', required=True,
                        help='Output file for training data (JSON)')

    parser.add_argument('--validation-file',
                        help='Output file for validation data (JSON). If not specified, no separate validation file is created.')

    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Ratio of validation examples (default: 0.1)')

    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process (default: all)')

    parser.add_argument('--max-chunk-size', type=int, default=2000,
                        help='Maximum chunk size in characters (default: 2000)')

    parser.add_argument('--example-count', type=int, default=2,
                        help='Number of examples to generate per chunk (default: 2)')

    parser.add_argument('--negative-count', type=int, default=2,
                        help='Number of negative examples per positive example (default: 2)')

    parser.add_argument('--batch-size', type=int, default=5,
                        help='Number of chunks to process in each batch (default: 5)')

    parser.add_argument('--model', default="gpt-4o-mini",
                        help='OpenAI model to use (default: gpt-4o-mini)')

    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for generation (default: 0.7)')

    parser.add_argument('--api-key',
                        help='OpenAI API key (default: from OPENAI_API_KEY environment variable)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    return parser.parse_args()


def main():
    """Main function to generate training data."""
    args = parse_arguments()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or use --api-key")
        return

    # Process input directory
    logger.info(f"Processing directory: {args.input_dir}")
    chunks = process_directory(
        args.input_dir,
        args.max_files,
        args.max_chunk_size
    )

    if not chunks:
        logger.error("No valid document chunks found. Check input directory.")
        return

    # Generate training examples
    logger.info(f"Generating training examples using {args.model}")
    examples = process_chunks(
        chunks,
        api_key,
        args.model,
        args.example_count,
        args.negative_count,
        args.batch_size,
        args.temperature,
        args.seed
    )

    if not examples:
        logger.error("Failed to generate training examples")
        return

    # Split into train and validation sets if needed
    if args.validation_file:
        train_data, val_data = split_train_val(examples, args.val_ratio)

        # Save training data
        save_to_file(train_data, args.output_file)

        # Save validation data
        save_to_file(val_data, args.validation_file)
    else:
        # Save all data as training
        save_to_file(examples, args.output_file)

    logger.info("Training data generation completed")


if __name__ == "__main__":
    main()