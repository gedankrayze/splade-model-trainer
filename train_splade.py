#!/usr/bin/env python3
"""
SPLADE Model Training Tool

This script fine-tunes a SPLADE (SParse Lexical AnD Expansion) model on domain-specific data.
It loads a pre-trained model and fine-tunes it using contrastive learning on query-document pairs.

Usage:
    python train_splade.py --train-file training_data.json --output-dir ./fine_tuned_splade
"""

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW, get_scheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_splade")


class SpladeDataset(Dataset):
    """Dataset for SPLADE model training with query-document pairs."""

    def __init__(self, data_file, tokenizer, max_length=512):
        """
        Initialize dataset from data file.

        Args:
            data_file: Path to JSON file with training data
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length for tokenizer
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        logger.info(f"Loading data from {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} training examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # Get query and positive document
        query = example["query"]
        positive_doc = example["positive_document"]

        # Get one random negative document
        negative_docs = example["negative_documents"]
        negative_doc = random.choice(negative_docs) if negative_docs else ""

        # Encode texts
        query_encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        positive_encoding = self.tokenizer(
            positive_doc,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        negative_encoding = self.tokenizer(
            negative_doc,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Convert to dict and remove batch dimension
        return {
            "query_input_ids": query_encoding["input_ids"].squeeze(0),
            "query_attention_mask": query_encoding["attention_mask"].squeeze(0),
            "positive_input_ids": positive_encoding["input_ids"].squeeze(0),
            "positive_attention_mask": positive_encoding["attention_mask"].squeeze(0),
            "negative_input_ids": negative_encoding["input_ids"].squeeze(0),
            "negative_attention_mask": negative_encoding["attention_mask"].squeeze(0),
        }


class SpladeTrainer:
    """Trainer for SPLADE model fine-tuning."""

    def __init__(
            self,
            model_name,
            output_dir,
            train_file,
            val_file=None,
            learning_rate=5e-5,
            batch_size=8,
            epochs=3,
            lambda_d=0.0001,  # Regularization for document vectors
            lambda_q=0.0001,  # Regularization for query vectors
            max_length=512,
            seed=42,
            device=None
    ):
        """
        Initialize SPLADE trainer.

        Args:
            model_name: Name or path of pre-trained model
            output_dir: Directory to save fine-tuned model
            train_file: Path to training data file
            val_file: Path to validation data file
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            lambda_d: Regularization coefficient for document vectors
            lambda_q: Regularization coefficient for query vectors
            max_length: Maximum sequence length
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.train_file = train_file
        self.val_file = val_file
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.max_length = max_length
        self.seed = seed

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get the device
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
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        # Move model to device
        self.model.to(self.device)

    def splade_pooling(self, logits, attention_mask=None):
        """
        Apply SPLADE pooling to get sparse representations.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Sparse representation [batch_size, vocab_size]
        """
        # Apply RELU to logits and then LOG(1 + x)
        # This is the SPLADE transformation
        activated = torch.log(1 + torch.relu(logits))

        # Apply max pooling over sequence dimension
        # This gives us one weight per vocabulary term
        if attention_mask is not None:
            # Mask out padding tokens
            mask = attention_mask.unsqueeze(-1).expand_as(activated)
            activated = activated * mask

        pooled = torch.max(activated, dim=1)[0]
        return pooled

    def compute_flops_loss(self, sparse_rep):
        """
        Compute FLOPS regularization loss.

        Args:
            sparse_rep: Sparse representation [batch_size, vocab_size]

        Returns:
            L1 norm of each sparse vector, summed
        """
        # Sum of L1 norms of sparse vectors
        return torch.sum(torch.norm(sparse_rep, p=1, dim=1))

    def train_epoch(self, train_loader, optimizer, scheduler):
        """
        Train model for one epoch.

        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer for parameter updates
            scheduler: Learning rate scheduler

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass for query
            query_outputs = self.model(
                input_ids=batch["query_input_ids"],
                attention_mask=batch["query_attention_mask"],
                return_dict=True
            )

            # Forward pass for positive document
            positive_outputs = self.model(
                input_ids=batch["positive_input_ids"],
                attention_mask=batch["positive_attention_mask"],
                return_dict=True
            )

            # Forward pass for negative document
            negative_outputs = self.model(
                input_ids=batch["negative_input_ids"],
                attention_mask=batch["negative_attention_mask"],
                return_dict=True
            )

            # Apply SPLADE pooling
            query_rep = self.splade_pooling(query_outputs.logits, batch["query_attention_mask"])
            positive_rep = self.splade_pooling(positive_outputs.logits, batch["positive_attention_mask"])
            negative_rep = self.splade_pooling(negative_outputs.logits, batch["negative_attention_mask"])

            # Compute similarity scores (dot products)
            positive_scores = torch.sum(query_rep * positive_rep, dim=1)
            negative_scores = torch.sum(query_rep * negative_rep, dim=1)

            # Compute contrastive loss (margin-based)
            margin = 0.2
            rank_loss = torch.mean(torch.relu(margin - positive_scores + negative_scores))

            # Compute regularization losses
            query_flops = self.compute_flops_loss(query_rep)
            doc_flops = self.compute_flops_loss(positive_rep)

            # Total loss
            loss = rank_loss + self.lambda_q * query_flops + self.lambda_d * doc_flops

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def evaluate(self, val_loader):
        """
        Evaluate model on validation data.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Average loss for validation data
        """
        self.model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass for query
                query_outputs = self.model(
                    input_ids=batch["query_input_ids"],
                    attention_mask=batch["query_attention_mask"],
                    return_dict=True
                )

                # Forward pass for positive document
                positive_outputs = self.model(
                    input_ids=batch["positive_input_ids"],
                    attention_mask=batch["positive_attention_mask"],
                    return_dict=True
                )

                # Forward pass for negative document
                negative_outputs = self.model(
                    input_ids=batch["negative_input_ids"],
                    attention_mask=batch["negative_attention_mask"],
                    return_dict=True
                )

                # Apply SPLADE pooling
                query_rep = self.splade_pooling(query_outputs.logits, batch["query_attention_mask"])
                positive_rep = self.splade_pooling(positive_outputs.logits, batch["positive_attention_mask"])
                negative_rep = self.splade_pooling(negative_outputs.logits, batch["negative_attention_mask"])

                # Compute similarity scores
                positive_scores = torch.sum(query_rep * positive_rep, dim=1)
                negative_scores = torch.sum(query_rep * negative_rep, dim=1)

                # Compute contrastive loss
                margin = 0.2
                rank_loss = torch.mean(torch.relu(margin - positive_scores + negative_scores))

                # Compute regularization losses
                query_flops = self.compute_flops_loss(query_rep)
                doc_flops = self.compute_flops_loss(positive_rep)

                # Total loss
                loss = rank_loss + self.lambda_q * query_flops + self.lambda_d * doc_flops

                batch_size = batch["query_input_ids"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        return avg_loss

    def train(self):
        """Train the SPLADE model."""
        # Create datasets
        logger.info("Creating datasets")
        train_dataset = SpladeDataset(
            self.train_file,
            self.tokenizer,
            max_length=self.max_length
        )

        val_dataset = None
        if self.val_file:
            val_dataset = SpladeDataset(
                self.val_file,
                self.tokenizer,
                max_length=self.max_length
            )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4
            )

        # Create optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # Create scheduler
        num_training_steps = self.epochs * len(train_loader)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )

        # Train model
        logger.info(f"Starting training for {self.epochs} epochs")
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")

            # Train epoch
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            logger.info(f"Training loss: {train_loss:.4f}")

            # Evaluate
            if val_loader:
                val_loss = self.evaluate(val_loader)
                logger.info(f"Validation loss: {val_loss:.4f}")

            # Save checkpoint
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save model
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)

            # Save training arguments
            with open(os.path.join(checkpoint_dir, "training_args.json"), "w") as f:
                json.dump({
                    "model_name": self.model_name,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "lambda_d": self.lambda_d,
                    "lambda_q": self.lambda_q,
                    "max_length": self.max_length,
                    "seed": self.seed
                }, f, indent=2)

        # Save final model
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # Save config for inference
        with open(os.path.join(self.output_dir, "splade_config.json"), "w") as f:
            json.dump({
                "model_type": "splade",
                "base_model_name": self.model_name,
                "max_length": self.max_length,
                "pooling": "max",
                "activation": "log1p_relu"
            },f, indent=2)

        logger.info(f"Training completed. Model saved to {self.output_dir}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train SPLADE model")

    parser.add_argument('--device', choices=['cuda', 'cpu', 'mps'], default=None,
                        help='Device to run model on (default: auto-detect)')

    parser.add_argument('--train-file', required=True,
                        help='Path to training data file (JSON)')

    parser.add_argument('--val-file',
                        help='Path to validation data file (JSON)')

    parser.add_argument('--output-dir', required=True,
                        help='Directory to save trained model')

    parser.add_argument('--model-name', default="prithivida/Splade_PP_en_v1",
                        help='Pre-trained model name or path (default: prithivida/Splade_PP_en_v1)')

    parser.add_argument('--learning-rate', type=float, default=5e-5,
                        help='Learning rate (default: 5e-5)')

    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')

    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')

    parser.add_argument('--lambda-d', type=float, default=0.0001,
                        help='Regularization coefficient for document vectors (default: 0.0001)')

    parser.add_argument('--lambda-q', type=float, default=0.0001,
                        help='Regularization coefficient for query vectors (default: 0.0001)')

    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length (default: 512)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Create trainer
    trainer = SpladeTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lambda_d=args.lambda_d,
        lambda_q=args.lambda_q,
        max_length=args.max_length,
        seed=args.seed,
        device=args.device
    )

    # Train model
    trainer.train()


if __name__ == "__main__":
    main()