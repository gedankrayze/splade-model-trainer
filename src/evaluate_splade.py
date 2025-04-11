#!/usr/bin/env python3
"""
SPLADE Model Evaluation Script

This script evaluates a trained SPLADE model on a test dataset,
calculating retrieval metrics like MRR, Precision@k, and NDCG.

Usage:
    python test_evaluate.py --model-dir ./fine_tuned_splade --test-file validation_data.json
"""

import argparse
import json
import logging
import os
import time
from typing import List, Dict, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("evaluate_splade")


class SpladeEvaluator:
    """Evaluator for SPLADE model."""

    def __init__(
            self,
            model_dir: str,
            max_length: int = 512,
            device: str = None
    ):
        """
        Initialize SPLADE evaluator.

        Args:
            model_dir: Directory containing trained model
            max_length: Maximum sequence length
            device: Device to run model on ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_dir = model_dir
        self.max_length = max_length

        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                logger.info("Using MPS (Metal Performance Shaders) for GPU acceleration.")
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                logger.info("Using CUDA for GPU acceleration.")
                self.device = torch.device("cuda")
            else:
                logger.info("Using CPU for inference.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load tokenizer and model
        logger.info(f"Loading model from: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Load SPLADE config if available
        self.config = {}
        config_path = os.path.join(model_dir, "splade_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
                logger.info(f"Loaded SPLADE config: {self.config}")

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into SPLADE sparse representation.

        Args:
            text: Text to encode

        Returns:
            Sparse vector representation
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        # Apply SPLADE pooling (log(1 + ReLU(x)))
        logits = outputs.logits
        activated = torch.log(1 + torch.relu(logits))

        # Max pooling over sequence dimension
        # This accounts for attention mask to ignore padding tokens
        attention_expanded = inputs["attention_mask"].unsqueeze(-1).expand_as(activated)
        masked_activated = activated * attention_expanded
        sparse_rep = torch.max(masked_activated, dim=1)[0]

        return sparse_rep.cpu().numpy()

    def batch_encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Encode multiple texts in batches.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding

        Returns:
            Array of sparse vector representations
        """
        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i + batch_size]

            # Process each text in batch individually to handle variable lengths
            batch_vectors = []
            for text in batch_texts:
                vector = self.encode_text(text)
                batch_vectors.append(vector)

            results.extend(batch_vectors)

        return np.vstack(results)

    def retrieve(
            self,
            query: str,
            documents: List[str],
            top_k: int = 10
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Retrieve top-k documents for a query.

        Args:
            query: Query text
            documents: List of documents to search
            top_k: Number of documents to retrieve

        Returns:
            List of dictionaries with document text and score
        """
        # Encode query
        query_vec = self.encode_text(query)

        # Encode documents (in batches)
        doc_vecs = self.batch_encode(documents)

        # Calculate scores using dot product
        scores = np.dot(query_vec, doc_vecs.T)[0]

        # Get top-k documents
        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": documents[idx],
                "score": float(scores[idx])
            })

        return results


def load_test_data(test_file: str) -> Tuple[List[str], List[str], List[List[str]]]:
    """
    Load test data from file.

    Args:
        test_file: Path to test file (JSON)

    Returns:
        Tuple of (queries, positive_docs, negative_docs)
    """
    logger.info(f"Loading test data from: {test_file}")

    with open(test_file, "r") as f:
        data = json.load(f)

    queries = []
    positive_docs = []
    negative_docs = []

    for item in data:
        queries.append(item["query"])
        positive_docs.append(item["positive_document"])

        # Handle different formats of negative documents
        item_negatives = []
        for neg_doc in item["negative_documents"]:
            if isinstance(neg_doc, dict) and "document" in neg_doc:
                item_negatives.append(neg_doc["document"])
            elif isinstance(neg_doc, str):
                item_negatives.append(neg_doc)

        negative_docs.append(item_negatives)

    logger.info(f"Loaded {len(queries)} queries with positive and negative documents")
    return queries, positive_docs, negative_docs


def evaluate(
        evaluator: SpladeEvaluator,
        queries: List[str],
        positive_docs: List[str],
        negative_docs: List[List[str]],
        k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate retrieval performance.

    Args:
        evaluator: SpladeEvaluator instance
        queries: List of query texts
        positive_docs: List of positive document texts
        negative_docs: List of lists of negative document texts
        k_values: Values of k for Precision@k metric

    Returns:
        Dictionary of metrics
    """
    metrics = {f"P@{k}": 0.0 for k in k_values}
    metrics.update({
        "MRR": 0.0,
        "NDCG@10": 0.0,
        "Recall@100": 0.0
    })

    # Encode all queries
    logger.info("Encoding queries...")
    query_vecs = evaluator.batch_encode(queries)

    total_time = 0
    n_queries = len(queries)

    for i, (query, pos_doc, neg_docs_list) in enumerate(zip(queries, positive_docs, negative_docs)):
        logger.info(f"Evaluating query {i + 1}/{n_queries}: {query}")

        # Combine positive and negative documents
        all_docs = [pos_doc] + neg_docs_list
        relevance = np.zeros(len(all_docs))
        relevance[0] = 1  # First document is the positive one

        # Encode documents
        doc_vecs = evaluator.batch_encode(all_docs)

        # Calculate scores
        start_time = time.time()
        scores = np.dot(query_vecs[i].reshape(1, -1), doc_vecs.T)[0]
        total_time += time.time() - start_time

        # Get ranking
        ranking = np.argsort(-scores)

        # Calculate metrics
        # MRR (Mean Reciprocal Rank)
        pos_rank = np.where(ranking == 0)[0][0] + 1  # +1 for 1-indexed rank
        metrics["MRR"] += 1.0 / pos_rank

        # Precision@k
        for k in k_values:
            if pos_rank <= k:
                metrics[f"P@{k}"] += 1.0

        # NDCG@10
        dcg = 0
        idcg = 1  # Ideal DCG with one relevant document at position 1
        for j, doc_idx in enumerate(ranking[:10]):
            if doc_idx == 0:  # If it's the positive document
                # Use 2^rel - 1 / log2(rank+1) formula for DCG
                dcg += (2 ** 1 - 1) / np.log2(j + 2)  # +2 for 1-indexed rank and log base 2

        metrics["NDCG@10"] += dcg / idcg

        # Recall@100
        if pos_rank <= 100:
            metrics["Recall@100"] += 1.0

    # Average metrics
    for key in metrics:
        metrics[key] /= n_queries

    # Add average query time
    metrics["Avg_Query_Time_ms"] = (total_time / n_queries) * 1000

    return metrics


def parse_arguments():
    """Parse command-line arguments."""
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) for GPU acceleration.")
        found_device = 'mps'
    elif torch.cuda.is_available():
        print("Using CUDA for GPU acceleration.")
        found_device = 'cuda'
    else:
        print("Using CPU for inference.")
        found_device = 'cpu'

    parser = argparse.ArgumentParser(description="Evaluate SPLADE model")

    parser.add_argument('--model-dir', required=True,
                        help='Directory containing trained model')

    parser.add_argument('--test-file', required=True,
                        help='Path to test data file (JSON)')

    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length (default: 512)')

    parser.add_argument('--device', choices=['cuda', 'cpu', 'mps'], default=None,
                        help='Device to run model on (default: auto-detect)')

    parser.add_argument('--output-file',
                        help='Path to save results (JSON)')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Initialize evaluator
    evaluator = SpladeEvaluator(
        model_dir=args.model_dir,
        max_length=args.max_length,
        device=args.device
    )

    # Load test data
    queries, positive_docs, negative_docs = load_test_data(args.test_file)

    # Evaluate model
    logger.info("Evaluating model...")
    metrics = evaluate(evaluator, queries, positive_docs, negative_docs)

    # Print results
    logger.info("Evaluation results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    # Save results
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to: {args.output_file}")

    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()