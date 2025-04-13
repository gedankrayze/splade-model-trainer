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
- Language support for multilingual data generation
"""

import argparse
import logging
import os
import sys
import asyncio
import random
from typing import List, Dict, Any, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("generate_training_data")

# Import generator modules
from src.generator.processors import process_directory
from src.generator.api import process_chunks_async, process_chunks
from src.generator.utils import split_train_val_test, save_to_file
from src.generator.templates import get_template, detect_document_language


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

    parser.add_argument('--domain-template', choices=['generic', 'technical', 'legal', 'medical', 'finance', 'multilingual'], 
                        default='generic',
                        help='Domain template to use (default: generic)')

    parser.add_argument('--language', 
                        help='Language for generated examples (e.g., "en", "de"). When specified, examples will be generated in this language')

    parser.add_argument('--detect-language', action='store_true',
                        help='Automatically detect document language and generate examples in same language')

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

    # Handle language detection if requested
    language = args.language
    if args.detect_language:
        logger.info("Attempting to auto-detect document language")
        # Use the 'multilingual' template which has auto-detection instructions
        args.domain_template = "multilingual"
        
        # Sample a few chunks to detect language
        sample_chunks = chunks[:min(5, len(chunks))]
        sample_text = "\n\n".join(chunk["content"] for chunk in sample_chunks)
        detected_language = detect_document_language(sample_text)
        
        if detected_language:
            logger.info(f"Detected document language: {detected_language}")
            language = detected_language
        else:
            logger.warning("Could not reliably detect document language, using multilingual template")

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
        language=language,
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
        try:
            # Import AsyncOpenAI to verify availability
            from openai import AsyncOpenAI
        except ImportError:
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

    # Handle language detection if requested
    language = args.language
    if args.detect_language:
        logger.info("Attempting to auto-detect document language")
        # Use the 'multilingual' template which has auto-detection instructions
        args.domain_template = "multilingual"
        
        # Sample a few chunks to detect language
        sample_chunks = chunks[:min(5, len(chunks))]
        sample_text = "\n\n".join(chunk["content"] for chunk in sample_chunks)
        detected_language = detect_document_language(sample_text)
        
        if detected_language:
            logger.info(f"Detected document language: {detected_language}")
            language = detected_language
        else:
            logger.warning("Could not reliably detect document language, using multilingual template")

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
        language=language,
        seed=args.seed
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
