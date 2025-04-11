#!/usr/bin/env python3
"""
Dense Embedding Model Training Tool

This script fine-tunes a dense embedding model on domain-specific data.
It loads a pre-trained model and fine-tunes it using contrastive learning on query-document pairs.

Usage:
    python train_embeddings.py --train-file training_data.json --output-dir ./fine_tuned_embeddings
"""

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AdamW, get_scheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_embeddings")


class EmbeddingDataset(Dataset):
    """Dataset for embedding model training with query-document pairs."""

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
        if isinstance(negative_docs[0], dict) and "document" in negative_docs[0]:
            # Handle case where negative_documents contains dictionaries
            negative_docs = [neg_doc["document"] for neg_doc in negative_docs]

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


class DenseEmbeddingTrainer:
    """Trainer for dense embedding model fine-tuning."""

    def __init__(
            self,
            model_name,
            output_dir,
            train_file,
            device=None,
            val_file=None,
            learning_rate=2e-5,
            batch_size=16,
            epochs=3,
            max_length=512,
            pooling_strategy="mean",  # mean, cls, or max
            temperature=0.1,  # Temperature for InfoNCE loss
            seed=42
    ):
        """
        Initialize embedding model trainer.

        Args:
            model_name: Name or path of pre-trained model
            output_dir: Directory to save fine-tuned model
            train_file: Path to training data file
            val_file: Path to validation data file
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            max_length: Maximum sequence length
            pooling_strategy: Pooling strategy for embeddings (mean, cls, max)
            temperature: Temperature parameter for contrastive loss
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.train_file = train_file
        self.val_file = val_file
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.temperature = temperature
        self.seed = seed

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        if device:
            # Get device
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Using device: {self.device}")

        # Load tokenizer and model
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Move model to device
        self.model.to(self.device)

    def get_embeddings(self, input_ids, attention_mask):
        """
        Get embeddings from model output using the specified pooling strategy.

        Args:
            input_ids: Input token ids
            attention_mask: Attention mask for tokens

        Returns:
            Normalized embeddings
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        # Get the hidden states
        hidden_states = outputs.last_hidden_state

        # Apply pooling based on the chosen strategy
        if self.pooling_strategy == "cls":
            # Use [CLS] token embedding
            embeddings = hidden_states[:, 0]
        elif self.pooling_strategy == "mean":
            # Mean pooling - take average of all token embeddings
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            embeddings = sum_embeddings / sum_mask
        elif self.pooling_strategy == "max":
            # Max pooling - take maximum of all token embeddings
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            # Replace padded tokens with large negative value
            hidden_states[mask_expanded == 0] = -1e9
            embeddings = torch.max(hidden_states, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Normalize embeddings to unit length
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings

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

            # Get embeddings
            query_embeddings = self.get_embeddings(
                batch["query_input_ids"],
                batch["query_attention_mask"]
            )

            positive_embeddings = self.get_embeddings(
                batch["positive_input_ids"],
                batch["positive_attention_mask"]
            )

            negative_embeddings = self.get_embeddings(
                batch["negative_input_ids"],
                batch["negative_attention_mask"]
            )

            # Compute similarity scores (dot products)
            positive_scores = torch.sum(query_embeddings * positive_embeddings, dim=1)
            negative_scores = torch.sum(query_embeddings * negative_embeddings, dim=1)

            # InfoNCE / NT-Xent loss
            # Concatenate positive and negative scores and apply temperature
            logits = torch.cat([positive_scores.unsqueeze(1), negative_scores.unsqueeze(1)], dim=1) / self.temperature

            # Labels are always 0 (positive example is at index 0)
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

            # Compute cross-entropy loss
            loss = F.cross_entropy(logits, labels)

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
            Average loss and accuracy for validation data
        """
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Get embeddings
                query_embeddings = self.get_embeddings(
                    batch["query_input_ids"],
                    batch["query_attention_mask"]
                )

                positive_embeddings = self.get_embeddings(
                    batch["positive_input_ids"],
                    batch["positive_attention_mask"]
                )

                negative_embeddings = self.get_embeddings(
                    batch["negative_input_ids"],
                    batch["negative_attention_mask"]
                )

                # Compute similarity scores
                positive_scores = torch.sum(query_embeddings * positive_embeddings, dim=1)
                negative_scores = torch.sum(query_embeddings * negative_embeddings, dim=1)

                # InfoNCE / NT-Xent loss
                logits = torch.cat([positive_scores.unsqueeze(1), negative_scores.unsqueeze(1)],
                                   dim=1) / self.temperature

                # Labels are always 0 (positive example is at index 0)
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

                # Compute cross-entropy loss
                loss = F.cross_entropy(logits, labels)

                # Count correct predictions (positive score > negative score)
                correct = (positive_scores > negative_scores).sum().item()

                batch_size = batch["query_input_ids"].size(0)
                total_loss += loss.item() * batch_size
                total_correct += correct
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def train(self):
        """Train the embedding model."""
        # Create datasets
        logger.info("Creating datasets")
        train_dataset = EmbeddingDataset(
            self.train_file,
            self.tokenizer,
            max_length=self.max_length
        )

        val_dataset = None
        if self.val_file:
            val_dataset = EmbeddingDataset(
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
                val_loss, val_acc = self.evaluate(val_loader)
                logger.info(f"Validation loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

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
                    "max_length": self.max_length,
                    "pooling_strategy": self.pooling_strategy,
                    "temperature": self.temperature,
                    "seed": self.seed
                }, f, indent=2)

        # Save final model
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # Save config for inference
        with open(os.path.join(self.output_dir, "embedding_config.json"), "w") as f:
            json.dump({
                "model_type": "dense_embedding",
                "base_model_name": self.model_name,
                "max_length": self.max_length,
                "pooling_strategy": self.pooling_strategy,
                "embedding_dim": self.model.config.hidden_size
            }, f, indent=2)

        logger.info(f"Training completed. Model saved to {self.output_dir}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train dense embedding model")

    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for training (e.g., "cuda", "mps", "cpu")')

    parser.add_argument('--train-file', required=True,
                        help='Path to training data file (JSON)')

    parser.add_argument('--val-file',
                        help='Path to validation data file (JSON)')

    parser.add_argument('--output-dir', required=True,
                        help='Directory to save trained model')

    parser.add_argument('--model-name', default="intfloat/e5-base-v2",
                        help='Pre-trained model name or path (default: intfloat/e5-base-v2)')

    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate (default: 2e-5)')

    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')

    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')

    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length (default: 512)')

    parser.add_argument('--pooling-strategy', choices=['mean', 'cls', 'max'], default='mean',
                        help='Pooling strategy for embeddings (default: mean)')

    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for contrastive loss (default: 0.1)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Create trainer
    trainer = DenseEmbeddingTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_length=args.max_length,
        pooling_strategy=args.pooling_strategy,
        temperature=args.temperature,
        seed=args.seed
    )

    # Train model
    trainer.train()


if __name__ == "__main__":
    main()
