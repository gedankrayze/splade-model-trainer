#!/usr/bin/env python3
"""
Memory-Optimized SPLADE Trainer Entry Point

This script provides a specialized entry point for training SPLADE models with
memory optimizations for systems with limited resources. It's particularly useful for:
- MacBooks with memory constraints in MPS (Metal Performance Shaders)
- Systems with limited GPU memory
- CPU-only training environments
- Training scenarios where memory efficiency is more important than speed

Key features:
- Explicit device selection (cuda, mps, cpu) with force option
- Implements gradient accumulation to reduce memory requirements
- Uses smaller batch sizes with more accumulation steps
- Includes memory tracking to avoid OOM errors
- Implements optional model parameter offloading

Example usage:
--------------
# Force CPU training with small batch size
python train_splade_optimized.py --train-file data/training.json --output-dir ./fine_tuned_splade \
    --device cpu --batch-size 2 --gradient-accumulation-steps 4

# Force MPS (Apple Silicon) training with memory tracking
python train_splade_optimized.py --train-file data/training.json --output-dir ./fine_tuned_splade \
    --device mps --memory-tracking --batch-size 4

# For extremely memory-constrained environments (any device)
python train_splade_optimized.py --train-file data/training.json --output-dir ./fine_tuned_splade \
    --batch-size 1 --gradient-accumulation-steps 8 --memory-efficient

For detailed documentation on the unified trainer, please see:
docs/unified_trainer.md
"""

import sys
import argparse
import gc
import logging
import os
import resource
import psutil
from typing import Optional, Dict, Any, Tuple

import torch
from src.unified.cli import parse_arguments as original_parse_arguments
from src.unified.trainer import UnifiedSpladeTrainer

# Import AdamW from torch.optim instead of transformers
from torch.optim import AdamW


def get_memory_usage() -> Tuple[float, float]:
    """
    Get current memory usage for tracking.
    
    Returns:
        A tuple of (RAM usage in GB, RAM percentage used)
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_gb = memory_info.rss / (1024 ** 3)  # Convert to GB
    memory_percent = process.memory_percent()
    
    return memory_usage_gb, memory_percent


def log_memory_usage(logger: logging.Logger) -> None:
    """
    Log current memory usage statistics.
    
    Args:
        logger: Logger to use for output
    """
    memory_gb, memory_percent = get_memory_usage()
    logger.info(f"Memory usage: {memory_gb:.2f} GB ({memory_percent:.1f}%)")
    
    # Get max memory usage from resource module (Linux/Mac only)
    try:
        max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS returns bytes, Linux returns kilobytes
        if sys.platform == 'darwin':
            max_memory = max_memory / (1024 * 1024)  # Convert to GB
        else:
            max_memory = max_memory / (1024)  # Convert to GB
        logger.info(f"Max memory usage (resource): {max_memory:.2f} GB")
    except Exception as e:
        logger.warning(f"Could not get max memory usage: {e}")


class OptimizedSpladeTrainer(UnifiedSpladeTrainer):
    """
    SPLADE trainer optimized for memory-constrained environments.
    
    This trainer extends the UnifiedSpladeTrainer with specialized methods for:
    1. Gradient accumulation to effectively increase batch size without increasing memory
    2. Optional memory tracking to avoid out-of-memory errors
    3. More efficient forward/backward passes with device-specific optimizations
    """
    
    def __init__(
            self,
            *args,
            gradient_accumulation_steps: int = 1,
            memory_tracking: bool = False,
            memory_efficient: bool = False,
            force_device: Optional[str] = None,
            **kwargs
    ):
        """
        Initialize memory-optimized SPLADE trainer.
        
        Args:
            *args: Arguments to pass to parent class
            gradient_accumulation_steps: Number of batches to accumulate gradients before updating weights
            memory_tracking: Whether to track and log memory usage
            memory_efficient: Use additional memory optimization techniques (parameter offloading, etc.)
            force_device: Force a specific device ('cuda', 'mps', 'cpu') regardless of availability
            **kwargs: Keyword arguments to pass to parent class
        """
        # Force specific device if requested
        if force_device:
            kwargs['device'] = force_device
            # Disable mixed precision when forcing CPU
            if force_device == 'cpu':
                kwargs['use_mixed_precision'] = False
        
        super().__init__(*args, **kwargs)
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.memory_tracking = memory_tracking
        self.memory_efficient = memory_efficient
        
        self.logger.info(f"Using memory-optimized trainer with gradient accumulation steps: {gradient_accumulation_steps}")
        if self.memory_tracking:
            self.logger.info(f"Memory tracking enabled on device: {self.device}")
            log_memory_usage(self.logger)
            
        if self.memory_efficient:
            self.logger.info("Memory-efficient mode enabled (parameter offloading, aggressive GC)")

    def train_epoch(self, train_loader, epoch):
        """
        Train model for one epoch with gradient accumulation and memory optimizations.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number (0-based)

        Returns:
            Average loss for the epoch
        """
        # Set model to training mode
        self.model.train()
        
        # Initialize metrics
        total_loss = 0
        rank_loss_total = 0
        query_flops_total = 0
        doc_flops_total = 0
        
        # Create progress bar with reduced update frequency for lower overhead
        from tqdm import tqdm
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{self.epochs}",
            mininterval=2.0  # Update less frequently (every 2 seconds)
        )
        
        # Batch counters
        batch_count = 0
        effective_batch_count = 0
        
        # Iterate through batches
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass and compute loss
                # Use mixed precision only if enabled and on CUDA
                mixed_precision = self.use_mixed_precision and self.device.type == 'cuda'
                
                if mixed_precision:
                    # Handle mixed precision training
                    with torch.cuda.amp.autocast():
                        outputs = self.forward_pass(batch, mixed_precision=True)
                        loss = outputs["loss"]
                        # Scale loss by gradient accumulation steps
                        scaled_loss = loss / self.gradient_accumulation_steps
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(scaled_loss).backward()
                else:
                    # Standard precision training
                    outputs = self.forward_pass(batch, mixed_precision=False)
                    loss = outputs["loss"]
                    # Scale loss by gradient accumulation steps
                    scaled_loss = loss / self.gradient_accumulation_steps
                    scaled_loss.backward()
                
                # Track metrics
                total_loss += loss.item()
                rank_loss_total += outputs["rank_loss"]
                query_flops_total += outputs["query_flops"]
                doc_flops_total += outputs["doc_flops"]
                
                # Increment batch counter
                batch_count += 1
                
                # Log memory usage if tracking is enabled
                if self.memory_tracking and batch_count % 10 == 0:
                    log_memory_usage(self.logger)
                
                # Only update weights after accumulating gradients
                if batch_count % self.gradient_accumulation_steps == 0:
                    # Update weights
                    if mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                    self.optimizer.zero_grad()
                    
                    # Step scheduler
                    if self.scheduler is not None:
                        self.scheduler.step()
                        
                    effective_batch_count += 1
                
                # Clean up memory for this batch iteration
                if self.memory_efficient:
                    # Explicitly delete outputs and free memory
                    del outputs
                    del loss
                    del scaled_loss
                    # Force garbage collection
                    if batch_count % 5 == 0:
                        gc.collect()
                
                # Update progress bar with current metrics every few batches
                if batch_count % 2 == 0:
                    progress_bar.set_postfix({
                        "loss": loss.item(),
                        "rank_loss": outputs["rank_loss"],
                        "q_flops": outputs["query_flops"],
                        "d_flops": outputs["doc_flops"],
                        "acc_step": batch_count % self.gradient_accumulation_steps
                    })
                
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_idx}: {e}")
                # Skip this batch but continue training
                continue
                
        # Handle any remaining accumulated gradients
        if batch_count % self.gradient_accumulation_steps != 0:
            if mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
                
            self.optimizer.zero_grad()
            effective_batch_count += 1
                
        # Calculate average metrics
        num_batches = batch_count  # Use actual number of batches processed
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_rank_loss = rank_loss_total / num_batches if num_batches > 0 else float('inf')
        avg_query_flops = query_flops_total / num_batches if num_batches > 0 else float('inf')
        avg_doc_flops = doc_flops_total / num_batches if num_batches > 0 else float('inf')
        
        # Log detailed metrics
        self.logger.info(f"  Rank Loss: {avg_rank_loss:.4f}")
        self.logger.info(f"  Query FLOPS Loss: {avg_query_flops:.4f}")
        self.logger.info(f"  Doc FLOPS Loss: {avg_doc_flops:.4f}")
        self.logger.info(f"  Batches: {batch_count} (Effective updates: {effective_batch_count})")
        
        # Force garbage collection at the end of epoch
        if self.memory_efficient:
            gc.collect()
            if self.memory_tracking:
                log_memory_usage(self.logger)
        
        return avg_loss

    def _create_optimizer_and_scheduler(self, num_training_steps):
        """
        Create optimizer with memory efficient settings.
        
        Args:
            num_training_steps: Total number of training steps
        """
        # Account for gradient accumulation steps in scheduler
        effective_training_steps = num_training_steps // self.gradient_accumulation_steps
        self.logger.info(f"Creating optimizer and scheduler for {effective_training_steps} effective steps "
                        f"(with {self.gradient_accumulation_steps} gradient accumulation steps)")
        
        # Create optimizer with memory efficient settings
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            # Memory efficient settings
            eps=1e-5,  # Slightly higher than default for better numerical stability
            weight_decay=0.01,
        )
        
        # Create scheduler with reduced warmup for faster progress
        warmup_steps = max(100, int(0.05 * effective_training_steps))  # Minimum 100 steps, or 5% of training
        
        self.logger.info(f"Creating learning rate scheduler with {warmup_steps} warmup steps "
                         f"out of {effective_training_steps} total effective steps")
        
        from transformers import get_scheduler
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=effective_training_steps
        )


def parse_arguments():
    """
    Parse command-line arguments for memory-optimized training.
    
    Returns:
        Namespace containing all parsed arguments
    """
    # Create new argument parser
    parser = argparse.ArgumentParser(
        description="Memory-Optimized SPLADE Trainer - For memory-constrained environments",
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
    
    hardware_group.add_argument('--force-device', action='store_true',
                       help='Force the specified device even if it might not be optimal')
    
    hardware_group.add_argument('--mixed-precision', action='store_true',
                       help='Enable mixed precision training for faster performance (requires CUDA)')
    
    hardware_group.add_argument('--fp16-opt-level', default="O1", choices=["O1", "O2", "O3"],
                       help='Mixed precision optimization level (O1 is recommended for most cases)')
    
    # Memory optimization arguments
    memory_group = parser.add_argument_group('Memory Optimization')
    memory_group.add_argument('--gradient-accumulation-steps', type=int, default=1,
                      help='Accumulate gradients over N steps (reduces memory usage)')
    
    memory_group.add_argument('--memory-tracking', action='store_true',
                      help='Track and log memory usage during training')
    
    memory_group.add_argument('--memory-efficient', action='store_true',
                      help='Use aggressive memory optimizations (slower but uses less RAM)')
    
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


def main():
    """
    Main function for the Memory-Optimized SPLADE Trainer.
    
    This function:
    1. Parses command-line arguments
    2. Sets up logging
    3. Creates the optimized trainer
    4. Starts the training process
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging with appropriate level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("splade_trainer_optimized")
    
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
    logger.info(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"  Memory tracking: {args.memory_tracking}")
    logger.info(f"  Memory efficient: {args.memory_efficient}")
    
    # Create trainer with a try-except block to handle initialization errors
    try:
        logger.info("Initializing optimized trainer...")
        
        # Create optimized trainer
        trainer = OptimizedSpladeTrainer(
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
            
            # Memory optimization parameters
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            memory_tracking=args.memory_tracking,
            memory_efficient=args.memory_efficient,
            force_device=args.device if args.force_device else None,
            
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
        elif isinstance(e, RuntimeError) and "MPS" in str(e):
            logger.error("MPS out of memory - try reducing batch size or switch to CPU with --device cpu")
        
        # Exit with error code
        sys.exit(1)


if __name__ == "__main__":
    main()
