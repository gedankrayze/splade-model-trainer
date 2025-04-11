#!/usr/bin/env python3
"""
Memory-Efficient Dense Embedding Model Training Tool

This script fine-tunes a dense embedding model on domain-specific data with optimizations
for lower memory usage. It uses gradient checkpointing and mixed precision training to reduce
memory requirements.

Usage:
    python train_embeddings_lowmem.py --train-file training_data.json --output-dir ./fine_tuned_embeddings --batch-size 2
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
logger = logging.getLogger("train_embeddings_lowmem")


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
    """Memory-efficient trainer for dense embedding model fine-tuning."""

    def __init__(
            self,
            model_name,
            output_dir,
            train_file,
            val_file=None,
            learning_rate=2e-5,
            batch_size=2,  # Reduced default batch size
            epochs=3,
            max_length=256,  # Reduced default max length
            pooling_strategy="mean",
            temperature=0.1,
            seed=42,
            use_gradient_checkpointing=True,  # Enable gradient checkpointing by default
            use_mixed_precision=True,  # Enable mixed precision by default
            accumulation_steps=4,  # Gradient accumulation for effective larger batch size
            device=None
    ):
        """
        Initialize embedding model trainer with memory optimizations.

        Args:
            model_name: Name or path of pre-trained model
            output_dir: Directory to save fine-tuned model
            train_file: Path to training data file
            val_file: Path to validation data file
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training (reduced for low memory)
            epochs: Number of training epochs
            max_length: Maximum sequence length
            pooling_strategy: Pooling strategy for embeddings (mean, cls, max)
            temperature: Temperature parameter for contrastive loss
            seed: Random seed for reproducibility
            use_gradient_checkpointing: Enable gradient checkpointing to save memory
            use_mixed_precision: Use mixed precision training (fp16) for GPU memory savings
            accumulation_steps: Number of steps to accumulate gradients
            device: Explicitly set device (cuda, cpu, mps or None for auto)
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
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_mixed_precision = use_mixed_precision
        self.accumulation_steps = accumulation_steps

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Set mixed precision availability
        self.can_use_mixed_precision = (self.device.type == 'cuda') and self.use_mixed_precision
        if self.use_mixed_precision and not self.can_use_mixed_precision:
            logger.warning("Mixed precision requested but not available on this device. Disabling.")

        # Load tokenizer and model
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Enable gradient checkpointing to save memory
        if self.use_gradient_checkpointing:
            logger.info("Enabling gradient checkpointing for memory efficiency")
            self.model.gradient_checkpointing_enable()

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
        Train model for one epoch with memory optimizations.

        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer for parameter updates
            scheduler: Learning rate scheduler

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        steps = 0
        
        # Set up scaler for mixed precision (if available)
        scaler = torch.cuda.amp.GradScaler() if self.can_use_mixed_precision else None
        
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Mixed precision context
            with torch.cuda.amp.autocast() if self.can_use_mixed_precision else torch.no_grad():
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
                logits = torch.cat([positive_scores.unsqueeze(1), negative_scores.unsqueeze(1)], dim=1) / self.temperature
                
                # Labels are always 0 (positive example is at index 0)
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
                
                # Compute cross-entropy loss
                loss = F.cross_entropy(logits, labels)
                
                # Scale loss by accumulation steps
                loss = loss / self.accumulation_steps
            
            # Backward pass with mixed precision handling
            if self.can_use_mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Update weights every accumulation_steps or at the end of epoch
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if self.can_use_mixed_precision:
                    # Unscale gradients and clip to avoid exploding gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Perform optimizer step and update scaler
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Normal optimization step
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                
                # Count as a full optimization step
                steps += 1

            # Update progress bar
            batch_loss = loss.item() * self.accumulation_steps  # Get the original loss value
            total_loss += batch_loss
            progress_bar.set_postfix({"loss": batch_loss})

        # Calculate average loss per optimization step
        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def evaluate(self, val_loader):
        """
        Evaluate model on validation data with memory optimizations.

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
                # Process in smaller batches if needed
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Use mixed precision for evaluation too to maintain consistency
                with torch.cuda.amp.autocast() if self.can_use_mixed_precision else torch.no_grad():
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
                    logits = torch.cat([positive_scores.unsqueeze(1), negative_scores.unsqueeze(1)], dim=1) / self.temperature
                    
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
        """Train the embedding model with memory optimizations."""
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
        # Note: we're using fewer workers to reduce memory pressure
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2  # Reduced number of workers
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2  # Reduced number of workers
            )

        # Create optimizer
        # Use a different optimizer setup for better memory efficiency
        # Split parameters into groups with different learning rates
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        # Create optimizer with parameter groups
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate
        )

        # Create scheduler
        # Adjust for gradient accumulation steps
        num_training_steps = self.epochs * (len(train_loader) // self.accumulation_steps)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )

        # Train model
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Gradient accumulation steps: {self.accumulation_steps}")
        logger.info(f"  Effective batch size: {self.batch_size * self.accumulation_steps}")
        logger.info(f"  Max sequence length: {self.max_length}")
        logger.info(f"  Gradient checkpointing: {self.use_gradient_checkpointing}")
        logger.info(f"  Mixed precision: {self.can_use_mixed_precision}")
        
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
                    "accumulation_steps": self.accumulation_steps,
                    "effective_batch_size": self.batch_size * self.accumulation_steps,
                    "epochs": self.epochs,
                    "max_length": self.max_length,
                    "pooling_strategy": self.pooling_strategy,
                    "temperature": self.temperature,
                    "gradient_checkpointing": self.use_gradient_checkpointing,
                    "mixed_precision": self.can_use_mixed_precision,
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
    parser = argparse.ArgumentParser(description="Train dense embedding model with memory optimizations")

    parser.add_argument('--train-file', required=True,
                        help='Path to training data file (JSON)')

    parser.add_argument('--val-file',
                        help='Path to validation data file (JSON)')

    parser.add_argument('--output-dir', required=True,
                        help='Directory to save trained model')

    parser.add_argument('--model-name', default="intfloat/e5-small-v2",
                        help='Pre-trained model name or path (default: intfloat/e5-small-v2)')

    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate (default: 2e-5)')

    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (default: 2)')

    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')

    parser.add_argument('--max-length', type=int, default=256,
                        help='Maximum sequence length (default: 256)')

    parser.add_argument('--pooling-strategy', choices=['mean', 'cls', 'max'], default='mean',
                        help='Pooling strategy for embeddings (default: mean)')

    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for contrastive loss (default: 0.1)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
                        
    parser.add_argument('--gradient-checkpointing', action='store_true', default=True,
                        help='Enable gradient checkpointing to save memory (default: True)')
                        
    parser.add_argument('--no-gradient-checkpointing', action='store_false', dest='gradient_checkpointing',
                        help='Disable gradient checkpointing')
                        
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Use mixed precision training if available (default: True)')
                        
    parser.add_argument('--no-mixed-precision', action='store_false', dest='mixed_precision',
                        help='Disable mixed precision training')
                        
    parser.add_argument('--accumulation-steps', type=int, default=4,
                        help='Number of steps to accumulate gradients (default: 4)')
                        
    parser.add_argument('--device', choices=['cuda', 'cpu', 'mps'], default=None,
                        help='Device to use for training (default: auto-detect)')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Create trainer with memory optimizations
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
        use_gradient_checkpointing=args.gradient_checkpointing,
        use_mixed_precision=args.mixed_precision,
        accumulation_steps=args.accumulation_steps,
        device=args.device,
        seed=args.seed
    )

    # Train model
    trainer.train()


if __name__ == "__main__":
    main()