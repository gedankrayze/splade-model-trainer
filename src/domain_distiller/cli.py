#!/usr/bin/env python3
"""
CLI entry point for the Domain Distiller tool.
Provides a unified interface for all domain distiller functionalities.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from .utils.logging_utils import setup_logging
from .bootstrapper import bootstrap_domain
from .query_generator import generate_queries
from .document_generator import generate_documents
from .validator import validate_dataset
from .formatter import format_dataset

logger = logging.getLogger(__name__)

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="domain_distiller",
        description="Generate domain-specific training data using LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument("--output-dir", "-o", type=str, default="./distilled_data",
                        help="Output directory for generated data")
    parser.add_argument("--api-base", type=str, default="https://api.openai.com/v1",
                        help="Base URL for OpenAI-compatible API")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (defaults to OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model name to use")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of async workers")
    
    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Bootstrap command
    bootstrap_parser = subparsers.add_parser(
        "bootstrap", help="Bootstrap domain knowledge"
    )
    bootstrap_parser.add_argument("--domain", type=str, required=True,
                                help="Domain specification (legal, medical, etc.)")
    bootstrap_parser.add_argument("--language", type=str, default="en",
                                help="Language (en, de, es, fr, etc.)")
    bootstrap_parser.add_argument("--concepts", type=int, default=50,
                                help="Number of domain concepts to generate")
    bootstrap_parser.add_argument("--template", type=str, default=None,
                                help="Use pre-configured domain template")
    
    # Generate queries command
    query_parser = subparsers.add_parser(
        "generate-queries", help="Generate domain-specific queries"
    )
    query_parser.add_argument("--domain-file", type=str, required=True,
                            help="Path to bootstrapped domain knowledge file")
    query_parser.add_argument("--count", type=int, default=100,
                            help="Number of queries to generate")
    query_parser.add_argument("--complexity", type=str, choices=["simple", "mixed", "complex"],
                            default="mixed", help="Query complexity level")
    
    # Generate documents command
    document_parser = subparsers.add_parser(
        "generate-documents", help="Generate answer documents for queries"
    )
    document_parser.add_argument("--queries-file", type=str, required=True,
                               help="Path to generated queries file")
    document_parser.add_argument("--positives-per-query", type=int, default=1,
                               help="Number of positive documents per query")
    document_parser.add_argument("--negatives-per-query", type=int, default=3,
                               help="Number of negative documents per query")
    document_parser.add_argument("--length", type=str, choices=["short", "medium", "long"],
                               default="medium", help="Document length")
    document_parser.add_argument("--contrastive", action="store_true",
                               help="Use contrastive pair generation for negative documents")
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate generated dataset"
    )
    validate_parser.add_argument("--dataset-file", type=str, required=True,
                               help="Path to dataset file to validate")
    validate_parser.add_argument("--strict", action="store_true",
                               help="Enable strict validation")
    
    # Format command
    format_parser = subparsers.add_parser(
        "format", help="Format dataset for different output types"
    )
    format_parser.add_argument("--dataset-file", type=str, required=True,
                             help="Path to dataset file to format")
    format_parser.add_argument("--format", type=str, choices=["splade", "json", "jsonl", "csv", "tsv"],
                             default="splade", help="Output format")
    format_parser.add_argument("--split", action="store_true",
                             help="Split into train/val/test sets")
    format_parser.add_argument("--train-ratio", type=float, default=0.8,
                             help="Training data ratio when splitting")
    format_parser.add_argument("--val-ratio", type=float, default=0.1,
                             help="Validation data ratio when splitting")
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run the full data generation pipeline"
    )
    pipeline_parser.add_argument("--domain", type=str, required=True,
                               help="Domain specification (legal, medical, etc.)")
    pipeline_parser.add_argument("--language", type=str, default="en",
                               help="Language (en, de, es, fr, etc.)")
    pipeline_parser.add_argument("--queries", type=int, default=100,
                               help="Number of queries to generate")
    pipeline_parser.add_argument("--template", type=str, default=None,
                               help="Use pre-configured domain template")
    pipeline_parser.add_argument("--format", type=str, choices=["splade", "json", "jsonl", "csv", "tsv"],
                               default="splade", help="Output format")
    pipeline_parser.add_argument("--contrastive", action="store_true",
                               help="Use contrastive pair generation for negative documents")
    pipeline_parser.add_argument("--split", action="store_true",
                               help="Split into train/val/test sets")
    
    return parser

def run_bootstrap(args: argparse.Namespace) -> None:
    """Run the domain knowledge bootstrapping."""
    logger.info(f"Bootstrapping domain knowledge for {args.domain} in {args.language}")
    bootstrap_domain(
        domain=args.domain,
        language=args.language,
        num_concepts=args.concepts,
        template_name=args.template,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output_dir,
        num_workers=args.workers
    )

def run_query_generation(args: argparse.Namespace) -> None:
    """Run the query generation."""
    logger.info(f"Generating {args.count} queries using domain knowledge from {args.domain_file}")
    generate_queries(
        domain_file=args.domain_file,
        count=args.count,
        complexity=args.complexity,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output_dir,
        num_workers=args.workers
    )

def run_document_generation(args: argparse.Namespace) -> None:
    """Run the document generation."""
    logger.info(f"Generating documents for queries from {args.queries_file}")
    generate_documents(
        queries_file=args.queries_file,
        positives_per_query=args.positives_per_query,
        negatives_per_query=args.negatives_per_query,
        length=args.length,
        contrastive=args.contrastive,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output_dir,
        num_workers=args.workers
    )

def run_validation(args: argparse.Namespace) -> None:
    """Run dataset validation."""
    logger.info(f"Validating dataset {args.dataset_file}")
    validate_dataset(
        dataset_file=args.dataset_file,
        strict=args.strict,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        num_workers=args.workers
    )

def run_formatting(args: argparse.Namespace) -> None:
    """Run dataset formatting."""
    logger.info(f"Formatting dataset {args.dataset_file} to {args.format} format")
    format_dataset(
        dataset_file=args.dataset_file,
        output_format=args.format,
        split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        output_dir=args.output_dir
    )

def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full pipeline."""
    logger.info(f"Running full pipeline for {args.domain} in {args.language}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Bootstrap domain
    domain_file = os.path.join(args.output_dir, f"{args.domain}_{args.language}_domain.json")
    bootstrap_domain(
        domain=args.domain,
        language=args.language,
        num_concepts=50,  # Default for pipeline
        template_name=args.template,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output_dir,
        output_file=domain_file,
        num_workers=args.workers
    )
    
    # 2. Generate queries
    queries_file = os.path.join(args.output_dir, f"{args.domain}_{args.language}_queries.json")
    generate_queries(
        domain_file=domain_file,
        count=args.queries,
        complexity="mixed",  # Default for pipeline
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output_dir,
        output_file=queries_file,
        num_workers=args.workers
    )
    
    # 3. Generate documents
    dataset_file = os.path.join(args.output_dir, f"{args.domain}_{args.language}_dataset.json")
    generate_documents(
        queries_file=queries_file,
        positives_per_query=1,  # Default for pipeline
        negatives_per_query=3,  # Default for pipeline
        length="medium",  # Default for pipeline
        contrastive=args.contrastive,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output_dir,
        output_file=dataset_file,
        num_workers=args.workers
    )
    
    # 4. Validate
    validate_dataset(
        dataset_file=dataset_file,
        strict=False,  # Default for pipeline
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        num_workers=args.workers
    )
    
    # 5. Format
    final_output = os.path.join(args.output_dir, f"{args.domain}_{args.language}_{args.format}")
    format_dataset(
        dataset_file=dataset_file,
        output_format=args.format,
        split=args.split,
        train_ratio=0.8,  # Default for pipeline
        val_ratio=0.1,  # Default for pipeline
        output_dir=args.output_dir,
        output_file=final_output
    )
    
    logger.info(f"Pipeline completed. Output saved to {args.output_dir}")

def main(args: Optional[List[str]] = None) -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Setup logging
    setup_logging(verbose=parsed_args.verbose)
    
    # Get API key from environment if not provided
    if not parsed_args.api_key:
        parsed_args.api_key = os.environ.get("OPENAI_API_KEY")
        if not parsed_args.api_key:
            logger.error("API key not provided. Use --api-key or set OPENAI_API_KEY environment variable.")
            sys.exit(1)
    
    # Create output directory
    if parsed_args.output_dir:
        os.makedirs(parsed_args.output_dir, exist_ok=True)
    
    # Execute requested command
    if parsed_args.command == "bootstrap":
        run_bootstrap(parsed_args)
    elif parsed_args.command == "generate-queries":
        run_query_generation(parsed_args)
    elif parsed_args.command == "generate-documents":
        run_document_generation(parsed_args)
    elif parsed_args.command == "validate":
        run_validation(parsed_args)
    elif parsed_args.command == "format":
        run_formatting(parsed_args)
    elif parsed_args.command == "pipeline":
        run_pipeline(parsed_args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
