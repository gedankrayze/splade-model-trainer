"""
Document processors for SPLADE training data generation.

This module provides functions for loading and processing documents for training data generation.
"""

import os
import re
import csv
import io
import json
import logging
import random
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from src.generator.models import DocumentChunk

# Configure logging
logger = logging.getLogger("generator.processors")


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
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
    from tqdm import tqdm
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
