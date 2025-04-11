#!/usr/bin/env python3
"""
Unified SPLADE Trainer Entry Point

This script serves as the main entry point for the unified SPLADE model trainer,
providing a convenient way to access all advanced training features through
a simple command-line interface.

SPLADE (SParse Lexical AnD Expansion) is a state-of-the-art approach to information
retrieval that combines the efficiency of sparse retrievers with the effectiveness
of neural language models. The SPLADE model creates sparse representations that
capture both lexical matching and semantic expansion.

This unified trainer brings together the best components from all previous
implementations into a cohesive, maintainable solution with features including:
- Mixed precision training for better performance
- Early stopping to prevent overfitting
- Checkpointing for saving/resuming training
- Detailed logging and metrics tracking
- Support for multiple hardware platforms (CUDA, MPS, CPU)

Example usage:
--------------
# Basic usage
python train_splade_unified.py --train-file data/training.json --output-dir ./fine_tuned_splade

# With validation and mixed precision
python train_splade_unified.py --train-file data/training.json --val-file data/validation.json \
    --output-dir ./fine_tuned_splade --mixed-precision

# With early stopping and custom hyperparameters
python train_splade_unified.py --train-file data/training.json --val-file data/validation.json \
    --output-dir ./fine_tuned_splade --early-stopping --learning-rate 3e-5 --batch-size 16

# See all available options
python train_splade_unified.py --help

For detailed documentation on the unified trainer, please see:
docs/unified_trainer.md
"""

import sys
from src.unified.cli import main

if __name__ == "__main__":
    """
    Entry point for the script.
    
    This imports and calls the main function from the CLI module, which:
    1. Parses command-line arguments
    2. Sets up logging
    3. Initializes the trainer
    4. Starts the training process
    """
    main()
