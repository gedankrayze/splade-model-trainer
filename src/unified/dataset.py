#!/usr/bin/env python3
"""
Unified SPLADE Dataset Module

This module provides the dataset implementation for SPLADE model training
with robust error handling and validation.

The dataset handles loading and preprocessing of training data in JSON format,
which contains query-document pairs used for contrastive learning.

Features:
- Robust error handling with detailed error messages
- Format validation to catch data issues early
- Automatic dataset structure analysis
- Support for different negative document formats
- Efficient tokenization and encoding for model input

Example data format expected:
[
    {
        "query": "example search query",
        "positive_document": "relevant document text",
        "negative_documents": ["irrelevant document 1", "irrelevant document 2", ...]
    },
    ...
]

Alternative negative documents format also supported:
[
    {
        "query": "example search query",
        "positive_document": "relevant document text",
        "negative_documents": [
            {"document": "irrelevant document 1"}, 
            {"document": "irrelevant document 2"},
            ...
        ]
    },
    ...
]
"""

import json
import logging
import random
from typing import Dict, Any, List, Optional, Union

import torch
from torch.utils.data import Dataset

from src.utils import DataError, FileSystemError, validate_file_exists


class SpladeDataset(Dataset):
    """
    Dataset for SPLADE model training with query-document pairs.
    
    This Dataset implementation is designed for contrastive learning with the SPLADE model.
    It loads training examples consisting of a query, a positive (relevant) document,
    and a list of negative (irrelevant) documents.
    
    For each example, during training, the dataset will return the query, the positive
    document, and one randomly selected negative document, all properly tokenized and
    encoded for the model.
    
    The dataset performs extensive validation to catch common issues with training data
    and provides detailed error information when problems are encountered.
    """

    def __init__(self, data_file: str, tokenizer: Any, max_length: int = 512, logger: Optional[logging.Logger] = None):
        """
        Initialize dataset from a JSON data file.

        Args:
            data_file: Path to JSON file with training data. The file should contain
                      a list of dictionaries, each with "query", "positive_document",
                      and "negative_documents" keys.
                      
            tokenizer: Tokenizer for encoding texts. Should be a Hugging Face tokenizer
                      compatible with the model being trained.
                      
            max_length: Maximum sequence length for tokenization. Longer sequences will
                       be truncated. Should match the model's limitations.
                       
            logger: Optional logger instance for recording dataset information
                   and issues. If None, a default logger will be created.
                   
        Raises:
            DataError: If the data file doesn't exist, can't be parsed, or has
                      invalid format
            FileSystemError: If there are issues accessing the data file
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.logger = logger or logging.getLogger(__name__)

        # Validate that file exists before attempting to read
        try:
            validate_file_exists(data_file, f"Training data file not found: {data_file}")
        except FileSystemError as e:
            raise DataError("Cannot load training data", e, {"file_path": data_file})

        # Load data with comprehensive error handling
        self.logger.info(f"Loading data from {data_file}")
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except json.JSONDecodeError as e:
            # Specific error for JSON parsing issues
            line_col = f"line {e.lineno}, column {e.colno}"
            raise DataError(f"Invalid JSON format in {data_file} at {line_col}: {e.msg}", 
                           e, {"file_path": data_file, "position": line_col})
        except UnicodeDecodeError as e:
            # Specific error for encoding issues
            raise DataError(f"Encoding error in {data_file}. File must be UTF-8 encoded.",
                           e, {"file_path": data_file})
        except Exception as e:
            # Generic catch-all for other file reading issues
            raise DataError(f"Error reading data file {data_file}", e, {"file_path": data_file})

        # Validate basic data format
        if not isinstance(self.data, list):
            raise DataError(
                f"Expected data to be a list, got {type(self.data).__name__}",
                context={'file_path': data_file}
            )

        # Log successful loading
        self.logger.info(f"Loaded {len(self.data)} training examples")
        
        # Analyze dataset structure and log statistics
        self._analyze_dataset()

    def _analyze_dataset(self) -> None:
        """
        Analyze the dataset structure and log statistics.
        
        This method examines a sample of the dataset to:
        1. Check for required keys in examples
        2. Identify the format of negative documents
        3. Calculate average query length
        4. Report potential issues
        
        This provides valuable insights about the dataset that can help
        identify issues early in the training process.
        """
        if not self.data:
            self.logger.warning("Dataset is empty")
            return
            
        # Sample a subset of examples to analyze (avoid analyzing entire dataset for large datasets)
        sample_size = min(5, len(self.data))
        samples = random.sample(self.data, sample_size)
        
        # Check for required keys in all samples
        required_keys = ["query", "positive_document"]
        missing_keys_count = 0
        
        for idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                self.logger.warning(f"Sample {idx} is not a dictionary")
                continue
                
            missing = [key for key in required_keys if key not in sample]
            if missing:
                missing_keys_count += 1
                
        if missing_keys_count > 0:
            self.logger.warning(f"{missing_keys_count}/{sample_size} samples are missing required keys")
            
        # Check for negative documents structure to understand what formats are present
        neg_docs_formats = set()
        for sample in samples:
            if isinstance(sample, dict) and "negative_documents" in sample:
                neg_docs = sample["negative_documents"]
                
                if isinstance(neg_docs, list):
                    if not neg_docs:
                        neg_docs_formats.add("empty_list")
                    elif isinstance(neg_docs[0], str):
                        neg_docs_formats.add("list_of_strings")
                    elif isinstance(neg_docs[0], dict) and "document" in neg_docs[0]:
                        neg_docs_formats.add("list_of_dicts_with_document_key")
                    else:
                        neg_docs_formats.add(f"list_of_{type(neg_docs[0]).__name__}")
                else:
                    neg_docs_formats.add(f"{type(neg_docs).__name__}")
                    
        self.logger.info(f"Negative documents formats detected: {', '.join(neg_docs_formats)}")
        
        # Log average query length for insight into the dataset
        if all(isinstance(sample, dict) and "query" in sample and isinstance(sample["query"], str) for sample in samples):
            avg_query_len = sum(len(sample["query"].split()) for sample in samples) / sample_size
            self.logger.info(f"Average query length: {avg_query_len:.1f} words")
            
            # Additional checks on very short or long queries
            short_queries = [s["query"] for s in samples if len(s["query"].split()) <= 2]
            long_queries = [s["query"] for s in samples if len(s["query"].split()) >= 20]
            
            if short_queries:
                self.logger.debug(f"Sample contains {len(short_queries)} very short queries (≤2 words)")
                
            if long_queries:
                self.logger.debug(f"Sample contains {len(long_queries)} very long queries (≥20 words)")
        
        # Calculate and log more dataset statistics from the sample
        if all(isinstance(sample, dict) for sample in samples):
            # Average positive document length
            if all("positive_document" in s and isinstance(s["positive_document"], str) for s in samples):
                avg_pos_len = sum(len(s["positive_document"].split()) for s in samples) / sample_size
                self.logger.info(f"Average positive document length: {avg_pos_len:.1f} words")
            
            # Average number of negative documents per example
            neg_docs_counts = []
            for s in samples:
                if "negative_documents" in s and isinstance(s["negative_documents"], list):
                    neg_docs_counts.append(len(s["negative_documents"]))
            
            if neg_docs_counts:
                avg_neg_count = sum(neg_docs_counts) / len(neg_docs_counts)
                min_neg = min(neg_docs_counts)
                max_neg = max(neg_docs_counts)
                self.logger.info(f"Negative documents per example: avg={avg_neg_count:.1f}, min={min_neg}, max={max_neg}")

    def __len__(self) -> int:
        """
        Get the number of examples in the dataset.
        
        Returns:
            Total number of examples in the dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example at the specified index.
        
        This method:
        1. Retrieves the example at the given index
        2. Validates the example format
        3. Extracts query, positive document, and a random negative document
        4. Tokenizes and encodes all texts
        5. Returns tensors ready for model input
        
        Args:
            idx: Index of the example to retrieve
            
        Returns:
            Dictionary with keys:
            - query_input_ids: Tensor of token IDs for query [seq_len]
            - query_attention_mask: Attention mask for query [seq_len]
            - positive_input_ids: Tensor of token IDs for positive document [seq_len]
            - positive_attention_mask: Attention mask for positive document [seq_len]
            - negative_input_ids: Tensor of token IDs for negative document [seq_len]
            - negative_attention_mask: Attention mask for negative document [seq_len]
            
        Raises:
            DataError: If there's an issue with the example format or processing
        """
        try:
            # Retrieve example at the specified index
            example = self.data[idx]

            # Validate example format is a dictionary
            if not isinstance(example, dict):
                raise DataError(
                    f"Expected example to be a dictionary, got {type(example).__name__}",
                    context={'example_idx': idx}
                )

            # Validate required keys are present
            required_keys = ["query", "positive_document"]
            missing_keys = [key for key in required_keys if key not in example]
            if missing_keys:
                raise DataError(
                    f"Missing required keys in example: {', '.join(missing_keys)}",
                    context={'example_idx': idx}
                )

            # Extract query and positive document
            query = example["query"]
            positive_doc = example["positive_document"]

            # Extract a random negative document, handling different formats
            negative_doc = ""
            if "negative_documents" in example:
                negative_docs = example["negative_documents"]
                if not negative_docs:
                    # Handle the case of empty negative documents list
                    self.logger.debug(f"Example {idx} has no negative documents, using empty string")
                else:
                    # Handle two common formats of negative documents
                    if isinstance(negative_docs[0], dict) and "document" in negative_docs[0]:
                        # Format: List of dictionaries with 'document' key
                        negative_docs = [nd["document"] for nd in negative_docs]
                    # Randomly select one negative document
                    negative_doc = random.choice(negative_docs)
            else:
                # Handle the case where negative_documents key is missing
                self.logger.debug(f"Example {idx} has no 'negative_documents' key, using empty string")

            # Encode query with tokenizer
            query_encoding = self.tokenizer(
                query,
                max_length=self.max_length,
                padding="max_length",  # Pad to max_length
                truncation=True,       # Truncate if longer than max_length
                return_tensors="pt"    # Return PyTorch tensors
            )

            # Encode positive document with tokenizer
            positive_encoding = self.tokenizer(
                positive_doc,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # Encode negative document with tokenizer
            negative_encoding = self.tokenizer(
                negative_doc,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # Remove batch dimension (squeeze) and return as dictionary
            return {
                # Query tensors
                "query_input_ids": query_encoding["input_ids"].squeeze(0),
                "query_attention_mask": query_encoding["attention_mask"].squeeze(0),
                
                # Positive document tensors
                "positive_input_ids": positive_encoding["input_ids"].squeeze(0),
                "positive_attention_mask": positive_encoding["attention_mask"].squeeze(0),
                
                # Negative document tensors
                "negative_input_ids": negative_encoding["input_ids"].squeeze(0),
                "negative_attention_mask": negative_encoding["attention_mask"].squeeze(0),
            }
        except Exception as e:
            # Log error and wrap in DataError with context information
            self.logger.error(f"Error processing example {idx}: {e}")
            raise DataError(f"Error processing example {idx}", e, {"example_idx": idx})
