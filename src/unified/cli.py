#!/usr/bin/env python3
"""
Command-line interface for the Unified SPLADE Trainer.

This module provides a comprehensive command-line interface for training SPLADE models
with all advanced features. It handles argument parsing, validation, and trainer initialization.

The CLI is designed to make it easy to train SPLADE models with different configurations
without having to modify code. All trainer parameters can be controlled via command-line
arguments with sensible defaults.

Example usage:
    # Basic usage with minimal arguments
    python train_splade_unified.py --train-file data/training.json --output-dir ./fine_tuned_splade
    
    # With validation and early stopping
    python train_splade_unified.py --train-file data/training.json --val-file data/validation.json \\
        --output-dir ./fine_tuned_splade --early-stopping
    
    # With mixed precision and custom hyperparameters
    python train_splade_unified.py --train-file data/training.json --val-file data/validation.json \\
        --output-dir ./fine_tuned_splade --mixed-precision --learning-rate 3e-5 --batch-size 16
        
    # Resuming from previous training
    python train_splade_unified.py --train-file data/training.json --val-file data/validation.json \\
        --output-dir ./fine_tuned_splade --resume-latest
"""

import argparse
import os
import sys
import logging
from typing import Any, Dict, Optional

import torch

from src.unified.trainer import UnifiedSpladeTrainer


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the Unified SPLADE Trainer.
    
    This function sets up all command-line arguments with appropriate defaults,
    help messages, and types. It uses ArgumentDefaultsHelpFormatter to 
    automatically include default values in help messages.
    
    Returns:
        Namespace containing all parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Unified SPLADE Trainer - Comprehensive toolkit for training SPLADE models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments group
    required_group = parser.add_argument_group('Required Arguments')
    required_group.add_argument('--train-file', required=True,
                       help='Path to training data file in JSON format containing query-document pairs')
    
    required_group.add_argument('--output-dir', required=True,
                       help='Directory to save trained model, checkpoints, and logs')
    
    # Dataset and model arguments
    data_group = parser.add_argument_group('Dataset and Model')
    data_group.add_argument('--val-file',
                       help='Path to validation data file in JSON format (enables early stopping and model selection)')
    
    data_group.add_argument('--model-name', default="prithivida/Splade_PP_en_v1",
                       help='Pre-trained model name from Hugging Face or local path')
    
    data_group.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length for tokenization (longer sequences will be truncated)')
    
    # Training hyperparameters
    training_group = parser.add_argument_group('Training Hyperparameters')
    training_group.add_argument('--learning-rate', type=float, default=5e-5,
                       help='Learning rate for optimizer')
    
    training_group.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training (reduce if encountering memory issues)')
    
    training_group.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (passes through the full dataset)')
    
    training_group.add_argument('--lambda-d', type=float, default=0.0001,
                       help='Regularization coefficient for document vectors (controls sparsity)')
    
    training_group.add_argument('--lambda-q', type=float, default=0.0001,
                       help='Regularization coefficient for query vectors (controls sparsity)')
    
    training_group.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
                       
    # Hardware acceleration
    hardware_group = parser.add_argument_group('Hardware Acceleration')
    hardware_group.add_argument('--device', choices=['cuda', 'cpu', 'mps'], default=None,
                       help='Device to run training on (auto-detected if not specified)')
    
    hardware_group.add_argument('--mixed-precision', action='store_true',
                       help='Enable mixed precision training for faster performance (requires CUDA)')
    
    hardware_group.add_argument('--fp16-opt-level', default="O1", choices=["O1", "O2", "O3"],
                       help='Mixed precision optimization level (O1 is recommended for most cases)')
    
    # Early stopping
    early_stopping_group = parser.add_argument_group('Early Stopping')
    early_stopping_group.add_argument('--early-stopping', action='store_true',
                       help='Enable early stopping to prevent overfitting (requires validation data)')
    
    early_stopping_group.add_argument('--early-stopping-patience', type=int, default=3,
                       help='Number of epochs with no improvement before stopping training')
    
    early_stopping_group.add_argument('--early-stopping-min-delta', type=float, default=0.0001,
                       help='Minimum change to qualify as improvement')
    
    early_stopping_group.add_argument('--early-stopping-monitor', default="val_loss",
                       help='Metric to monitor for early stopping (e.g., val_loss, train_loss)')
    
    early_stopping_group.add_argument('--early-stopping-mode', choices=['min', 'max'], default="min",
                       help='Whether to minimize or maximize the monitored metric')
    
    # Checkpointing
    checkpoint_group = parser.add_argument_group('Checkpointing')
    checkpoint_group.add_argument('--save-best-only', action='store_true',
                       help='Save only the best model based on monitored metric (saves disk space)')
    
    checkpoint_group.add_argument('--save-freq', type=int, default=1,
                       help='Save checkpoint every N epochs')
    
    checkpoint_group.add_argument('--max-checkpoints', type=int, default=3,
                       help='Maximum number of checkpoints to keep (removes oldest)')
    
    # Training recovery
    recovery_group = parser.add_argument_group('Training Recovery')
    recovery_group.add_argument('--resume-from-checkpoint',
                       help='Path to specific checkpoint to resume training from')
    
    recovery_group.add_argument('--resume-latest', action='store_true',
                       help='Resume from the latest checkpoint in output directory')
    
    recovery_group.add_argument('--resume-best', action='store_true',
                       help='Resume from the best checkpoint in output directory')
    
    # Logging
    logging_group = parser.add_argument_group('Logging')
    logging_group.add_argument('--log-dir',
                       help='Directory for logs (defaults to output_dir/logs)')
    
    logging_group.add_argument('--verbose', action='store_true',
                       help='Enable verbose (debug) logging with more detailed information')

    return parser.parse_args()


def main() -> None:
    """
    Main function for the Unified SPLADE Trainer CLI.
    
    This function:
    1. Parses command-line arguments
    2. Sets up logging
    3. Validates argument combinations and warns about potential issues
    4. Creates and initializes the trainer
    5. Starts the training process
    6. Handles any exceptions that may occur during training
    
    Returns:
        None
    
    Exits:
        0 on successful completion
        1 on error
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up CLI-specific logger without affecting root logger
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Create a CLI-specific logger
    logger = logging.getLogger("splade_trainer_cli")
    logger.setLevel(log_level)
    
    # Only add handler if none exist
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    # Check if validation file is provided when using features that require it
    if not args.val_file:
        if args.early_stopping and args.early_stopping_monitor == "val_loss":
            logger.warning("Early stopping with val_loss monitor requires a validation file. "
                          "Disabling early stopping.")
            args.early_stopping = False
        
        if args.save_best_only and args.early_stopping_monitor == "val_loss":
            logger.warning("Saving best model based on val_loss requires a validation file. "
                          "Disabling save-best-only.")
            args.save_best_only = False
    
    # Check that resume options are mutually exclusive
    resume_options = sum([
        args.resume_from_checkpoint is not None,
        args.resume_latest,
        args.resume_best
    ])
    if resume_options > 1:
        logger.error("Only one resume option can be specified: --resume-from-checkpoint, --resume-latest, or --resume-best")
        sys.exit(1)
        
    # Log key configuration parameters
    logger.info(f"Training configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Train file: {args.train_file}")
    logger.info(f"  Validation file: {args.val_file or 'None'}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Device: {args.device or 'auto-detect'}")
    logger.info(f"  Mixed precision: {args.mixed_precision}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    
    # Create trainer with a try-except block to handle initialization errors
    try:
        logger.info("Initializing trainer...")
        trainer = UnifiedSpladeTrainer(
            # Model and data parameters
            model_name=args.model_name,
            output_dir=args.output_dir,
            train_file=args.train_file,
            val_file=args.val_file,
            max_length=args.max_length,
            
            # Training hyperparameters
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lambda_d=args.lambda_d,
            lambda_q=args.lambda_q,
            seed=args.seed,
            
            # Hardware acceleration
            device=args.device,
            use_mixed_precision=args.mixed_precision,
            fp16_opt_level=args.fp16_opt_level,
            
            # Logging configuration
            log_dir=args.log_dir,
            
            # Early stopping parameters
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            early_stopping_monitor=args.early_stopping_monitor,
            early_stopping_mode=args.early_stopping_mode,
            
            # Checkpointing parameters
            save_best_only=args.save_best_only,
            save_freq=args.save_freq,
            max_checkpoints=args.max_checkpoints,
            
            # Recovery parameters
            resume_from_checkpoint=args.resume_from_checkpoint,
            resume_latest=args.resume_latest,
            resume_best=args.resume_best
        )
        
        # Start training with a try-except block to catch training errors
        logger.info("Starting training...")
        trainer.train()
        
        # Log success message
        logger.info("Training completed successfully")
        
    except KeyboardInterrupt:
        # Handle user interruption
        logger.info("Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        # Log error details with stack trace for debugging
        logger.error(f"Training failed: {e}", exc_info=True)
        
        # Specific error handling based on error type
        if isinstance(e, FileNotFoundError):
            logger.error("Check that all file paths are correct and files exist")
        elif isinstance(e, torch.cuda.OutOfMemoryError):
            logger.error("CUDA out of memory - try reducing batch size or model size")
        elif isinstance(e, RuntimeError) and "CUDA" in str(e):
            logger.error("CUDA error - check your GPU setup and drivers")
        
        # Exit with error code
        sys.exit(1)


if __name__ == "__main__":
    main()
