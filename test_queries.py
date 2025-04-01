#!/usr/bin/env python3
"""
SPLADE Model Query Testing Script

This script allows you to interactively test a trained SPLADE model with custom queries
against a collection of documents.

Usage:
    python test_queries.py --model-dir ./fine_tuned_splade --docs-file documents.json
"""

import argparse
import json
import logging
import sys
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_queries")


def load_splade_model(model_dir: str):
    """
    Load SPLADE model from directory.

    Args:
        model_dir: Directory containing the model

    Returns:
        Tuple of (tokenizer, model, device)
    """
    # Set device
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Metal Performance Shaders) for GPU acceleration.")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA for GPU acceleration.")
        device = torch.device("cuda")
    else:
        logger.info("Using CPU for inference.")
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    logger.info(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    return tokenizer, model, device


def encode_text(text: str, tokenizer, model, device, max_length: int = 512):
    """
    Encode text into SPLADE sparse representation.

    Args:
        text: Text to encode
        tokenizer: Tokenizer
        model: Model
        device: Device to run on
        max_length: Maximum sequence length

    Returns:
        Sparse vector representation
    """
    # Tokenize
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model output
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)

    # Apply SPLADE pooling (log(1 + ReLU(x)))
    logits = outputs.logits
    activated = torch.log(1 + torch.relu(logits))

    # Max pooling over sequence dimension
    attention_expanded = inputs["attention_mask"].unsqueeze(-1).expand_as(activated)
    masked_activated = activated * attention_expanded
    sparse_rep = torch.max(masked_activated, dim=1)[0]

    return sparse_rep.cpu().numpy()


def load_documents(docs_file: str):
    """
    Load documents from file.

    Args:
        docs_file: Path to document file (JSON)

    Returns:
        List of documents
    """
    logger.info(f"Loading documents from: {docs_file}")

    with open(docs_file, "r") as f:
        # Try to determine format
        data = json.load(f)

        # Check if it's a list of strings
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            documents = data
        # Check if it's a list of dicts with 'content' field
        elif isinstance(data, list) and all(isinstance(item, dict) and 'content' in item for item in data):
            documents = [item['content'] for item in data]
        # Check if it's a training data format with positive_document field
        elif isinstance(data, list) and all(isinstance(item, dict) and 'positive_document' in item for item in data):
            # Collect unique documents from both positive and negative examples
            documents = set()
            for item in data:
                documents.add(item['positive_document'])
                for neg_doc in item['negative_documents']:
                    if isinstance(neg_doc, dict) and 'document' in neg_doc:
                        documents.add(neg_doc['document'])
                    elif isinstance(neg_doc, str):
                        documents.add(neg_doc)
            documents = list(documents)
        else:
            logger.error(f"Unsupported document format in {docs_file}")
            sys.exit(1)

    logger.info(f"Loaded {len(documents)} documents")
    return documents


def retrieve_documents(query: str, documents: List[str], tokenizer, model, device, top_k: int = 5):
    """
    Retrieve top-k documents for a query.

    Args:
        query: Query text
        documents: List of documents to search
        tokenizer: Tokenizer
        model: Model
        device: Device to run on
        top_k: Number of documents to retrieve

    Returns:
        List of dictionaries with document indices, text snippets, and scores
    """
    # Encode query
    query_vec = encode_text(query, tokenizer, model, device)

    # Encode documents
    logger.info("Encoding documents...")
    doc_vecs = np.vstack([
        encode_text(doc, tokenizer, model, device)
        for doc in tqdm(documents)
    ])

    # Calculate scores
    scores = np.dot(query_vec, doc_vecs.T)[0]

    # Get top-k documents
    top_indices = np.argsort(-scores)[:top_k]

    results = []
    for i, idx in enumerate(top_indices):
        # Create a snippet (first 200 characters)
        doc_text = documents[idx]
        snippet = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text

        results.append({
            "rank": i + 1,
            "index": int(idx),
            "score": float(scores[idx]),
            "snippet": snippet
        })

    return results


def interactive_query(tokenizer, model, device, documents: List[str]):
    """
    Run interactive query loop.

    Args:
        tokenizer: Tokenizer
        model: Model
        device: Device to run on
        documents: List of documents to search
    """
    print("\n===== SPLADE Model Interactive Query =====")
    print(f"Loaded {len(documents)} documents for searching")
    print("Enter 'q' or 'quit' to exit")

    while True:
        print("\nEnter your query:")
        query = input("> ").strip()

        if query.lower() in ('q', 'quit', 'exit'):
            break

        if not query:
            continue

        print(f"\nSearching for: '{query}'")
        results = retrieve_documents(query, documents, tokenizer, model, device)

        print("\nResults:")
        for result in results:
            print(f"Rank {result['rank']} (Score: {result['score']:.4f}):")
            print(f"  {result['snippet']}")
            print()

        # Ask if user wants to see full document
        print("Enter document rank number to see full text (or press Enter to continue)")
        choice = input("> ").strip()

        if choice and choice.isdigit():
            rank = int(choice)
            for result in results:
                if result["rank"] == rank:
                    print("\nFull Document Text:")
                    print("=" * 80)
                    print(documents[result["index"]])
                    print("=" * 80)
                    break


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test SPLADE model with queries")

    parser.add_argument('--model-dir', required=True,
                        help='Directory containing trained model')

    parser.add_argument('--docs-file', required=True,
                        help='Path to documents file (JSON)')

    parser.add_argument('--query',
                        help='Single query to test (if not provided, enters interactive mode)')

    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of documents to retrieve (default: 5)')

    parser.add_argument('--output-file',
                        help='Path to save results for single query (JSON)')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Load model
    tokenizer, model, device = load_splade_model(args.model_dir)

    # Load documents
    documents = load_documents(args.docs_file)

    # Either run single query or interactive mode
    if args.query:
        logger.info(f"Searching for query: {args.query}")
        results = retrieve_documents(
            args.query, documents, tokenizer, model, device, args.top_k
        )

        # Print results
        print("\nResults:")
        for result in results:
            print(f"Rank {result['rank']} (Score: {result['score']:.4f}):")
            print(f"  {result['snippet']}")
            print()

        # Save results
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.output_file}")

    else:
        # Interactive mode
        interactive_query(tokenizer, model, device, documents)


if __name__ == "__main__":
    main()
