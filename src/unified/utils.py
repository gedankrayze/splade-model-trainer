#!/usr/bin/env python3
"""
Unified Utilities for SPLADE Trainer

This module provides utility functions and classes for the unified SPLADE trainer
including logging, error handling, checkpointing, and early stopping.
"""

import copy
import functools
import glob
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional, Callable, List, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer


# --- Custom JSON Encoder for PyTorch Tensors ---

class TensorJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles PyTorch Tensors.
    
    This encoder converts Tensors to Python lists or scalar values
    that can be safely serialized to JSON.
    """
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            # Handle different tensor shapes
            if obj.numel() == 1:  # Single value tensor
                return obj.item()  # Convert to Python scalar
            else:
                return obj.tolist()  # Convert to Python list
        # Let the parent class handle other types
        return super().default(obj)


# --- Validation Utilities ---

def validate_dir_exists(path: str, create: bool = False) -> None:
    """
    Validate that a directory exists, optionally creating it.
    
    Args:
        path: Directory path
        create: Whether to create the directory if it doesn't exist
    
    Raises:
        FileSystemError: If directory doesn't exist and create=False or 
                         if directory creation fails
    """
    if not os.path.exists(path):
        if create:
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                from src.utils import FileSystemError
                raise FileSystemError(f"Failed to create directory: {path}", e, {"path": path})
        else:
            from src.utils import FileSystemError
            raise FileSystemError(f"Directory does not exist: {path}", None, {"path": path})


def validate_file_exists(path: str, error_message: str = None) -> None:
    """
    Validate that a file exists.
    
    Args:
        path: File path
        error_message: Optional custom error message
    
    Raises:
        FileSystemError: If file doesn't exist
    """
    if not os.path.isfile(path):
        from src.utils import FileSystemError
        message = error_message or f"File does not exist: {path}"
        raise FileSystemError(message, None, {"path": path})


# --- Decorator Utilities ---

def catch_and_log_exceptions(max_attempts: int = 1, delay: float = 0.0, exceptions: tuple = Exception):
    """
    Decorator that catches exceptions, logs them, and optionally retries.
    
    Args:
        max_attempts: Maximum number of attempts (1 = no retry)
        delay: Delay between attempts in seconds
        exceptions: Tuple of exception types to catch
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger from self if it's a method, or create one
            logger = getattr(args[0], 'logger', logging.getLogger()) if args else logging.getLogger()
            
            attempt = 0
            while attempt < max_attempts:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.error(f"Exception in {func.__name__}: {e}", exc_info=True)
                    if attempt < max_attempts:
                        logger.info(f"Retrying in {delay} seconds... (Attempt {attempt}/{max_attempts})")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
                        raise
        return wrapper
    return decorator


# --- Logging Utilities ---

class TrainingLogger:
    """Enhanced logging for model training with metrics tracking."""
    
    def __init__(self, name: str, log_dir: str):
        """
        Initialize the training logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
        """
        self.name = name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        
        # Track existing handlers to avoid adding duplicates
        existing_handlers = {}
        for handler in self.logger.handlers:
            # Identify handler by its formatter and class name
            if hasattr(handler, 'formatter'):
                handler_key = (handler.formatter._fmt, handler.__class__.__name__)
                existing_handlers[handler_key] = True
        
        # Only reset the handlers if this specific logger doesn't seem to be initialized yet
        if not existing_handlers:
            # Set the root logger level
            self.logger.setLevel(logging.INFO)
            
            # Create console handler if not already present
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler_key = (console_formatter._fmt, 'StreamHandler')
            
            if console_handler_key not in existing_handlers:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
            
            # Create file handler if not already present
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"training_{current_time}.log")
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler_key = (file_formatter._fmt, 'FileHandler')
            
            if file_handler_key not in existing_handlers:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
        
        # Dictionary to store training metrics
        self.metrics = {
            "config": {},
            "epochs": {},
            "summary": {}
        }
        self.start_time = None
        
    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
        
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)
        
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
        
    def error(self, message: str, exc_info: bool = False) -> None:
        """Log an error message."""
        self.logger.error(message, exc_info=exc_info)
        
    def start_training(self, config: Dict[str, Any]) -> None:
        """Log the start of training and store configuration."""
        self.start_time = time.time()
        self.metrics["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.metrics["config"] = config
        
        # Log some important config details
        self.info(f"Training started")
        self.info(f"Training configuration: {json.dumps(config, indent=2, cls=TensorJSONEncoder)}")
        
    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for an epoch."""
        self.metrics["epochs"][epoch] = metrics
        self.info(f"Epoch {epoch} completed")
        self.info(f"Metrics: {json.dumps(metrics, indent=2, cls=TensorJSONEncoder)}")
        
    def end_training(self, summary: Dict[str, Any]) -> None:
        """Log the end of training and store summary metrics."""
        duration = time.time() - self.start_time if self.start_time else 0
        self.metrics["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.metrics["duration_seconds"] = duration
        self.metrics["summary"] = summary
        
        self.info("Training completed")
        self.info(f"Total duration: {duration:.2f} seconds")
        self.info(f"Summary: {json.dumps(summary, indent=2, cls=TensorJSONEncoder)}")
            
    def save_metrics(self) -> None:
        """Save all metrics to a JSON file."""
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(self.log_dir, f"metrics_{current_time}.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, cls=TensorJSONEncoder)
            
        self.info(f"Saved metrics to {metrics_file}")


# --- Training Utilities ---

class EarlyStopping:
    """
    Early stopping implementation to prevent overfitting.
    Monitors a metric and stops training if no improvement is seen
    for a number of epochs.
    """
    
    def __init__(
            self, 
            monitor: str = "val_loss", 
            min_delta: float = 0.0, 
            patience: int = 0,
            mode: str = "min",
            verbose: bool = False,
            logger: Optional[Any] = None
    ):
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor.
            min_delta: Minimum change to qualify as improvement.
            patience: Number of epochs with no improvement after which training will stop.
            mode: 'min' or 'max' (whether to minimize or maximize the monitored metric).
            verbose: Whether to print messages.
            logger: Optional logger.
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.logger = logger or logging.getLogger(__name__)
        
        self.best_epoch = 0
        self.counter = 0
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.stop_training = False
        
        self.logger.info(f"Early stopping initialized (monitor={monitor}, patience={patience}, mode={mode})")
        
    def __call__(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop.
        
        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metrics.
            
        Returns:
            True if training should stop, False otherwise.
        """
        if self.monitor not in metrics:
            if self.verbose:
                self.logger.warning(f"Early stopping metric '{self.monitor}' is not available in metrics")
            return False
            
        current = metrics[self.monitor]
        
        # Convert tensor to float if needed
        if isinstance(current, torch.Tensor):
            current = current.item()
        
        if self.mode == "min":
            # Check if current value is better (lower) than best value
            if current < self.best_value - self.min_delta:
                self._improvement(epoch, current)
            else:
                self._no_improvement()
        else:
            # Check if current value is better (higher) than best value
            if current > self.best_value + self.min_delta:
                self._improvement(epoch, current)
            else:
                self._no_improvement()
                
        return self.stop_training
        
    def _improvement(self, epoch: int, current: float) -> None:
        """Handle improvement in the monitored metric."""
        self.best_value = current
        self.best_epoch = epoch
        self.counter = 0
        if self.verbose:
            self.logger.info(f"Early stopping: improvement in {self.monitor} to {current:.6f}")
            
    def _no_improvement(self) -> None:
        """Handle no improvement in the monitored metric."""
        self.counter += 1
        if self.verbose:
            self.logger.info(f"Early stopping: no improvement for {self.counter}/{self.patience} epochs")
            
        if self.counter >= self.patience:
            self.stop_training = True
            if self.verbose:
                self.logger.info(f"Early stopping triggered after {self.counter} epochs with no improvement")


class Checkpointing:
    """
    Handles model checkpointing during training with options to save:
    - Periodically (every N epochs)
    - Best models only (based on validation metric)
    - Limited number of checkpoints (removing older ones)
    """
    
    def __init__(
            self,
            output_dir: str,
            save_best_only: bool = False,
            monitor: str = "val_loss",
            mode: str = "min",
            verbose: bool = False,
            save_freq: Optional[int] = None,
            max_checkpoints: Optional[int] = None,
            logger: Optional[Any] = None
    ):
        """
        Initialize checkpointing.
        
        Args:
            output_dir: Directory to save checkpoints.
            save_best_only: Whether to save only the best model.
            monitor: Metric to monitor for determining the best model.
            mode: 'min' or 'max' (whether to minimize or maximize the monitored metric).
            verbose: Whether to print messages.
            save_freq: Save checkpoint every N epochs (None = every epoch).
            max_checkpoints: Maximum number of checkpoints to keep (None = keep all).
            logger: Optional logger.
        """
        self.output_dir = output_dir
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.save_freq = save_freq
        self.max_checkpoints = max_checkpoints
        self.logger = logger or logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.best_path = None
        
        self.logger.info(f"Checkpointing initialized (dir={output_dir}, save_best_only={save_best_only})")
        
    def save(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            epoch: int,
            metrics: Dict[str, float],
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[Any] = None
    ) -> Optional[str]:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save.
            tokenizer: Tokenizer to save.
            epoch: Current epoch number.
            metrics: Dictionary of metrics.
            optimizer: Optional optimizer to save state.
            scheduler: Optional scheduler to save state.
            
        Returns:
            Path to saved checkpoint or None if not saved.
        """
        # Check if we should save based on frequency
        if self.save_freq is not None and epoch % self.save_freq != 0:
            return None
            
        # Check if we should save based on metric
        if self.save_best_only and self.monitor in metrics:
            current = metrics[self.monitor]
            
            # Convert tensor to float if needed
            if isinstance(current, torch.Tensor):
                current = current.item()
            
            # Determine if current model is the best
            is_best = False
            if self.mode == "min":
                if current < self.best_value:
                    self.best_value = current
                    is_best = True
            else:
                if current > self.best_value:
                    self.best_value = current
                    is_best = True
                    
            if not is_best:
                return None
                
        # Create checkpoint path
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{epoch:03d}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        # Save optimizer and scheduler states if provided
        training_state = {}
        if optimizer is not None:
            # We need to handle tensors in the optimizer state
            # Just save the state_dict path, we'll save the actual state separately
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            torch.save(optimizer.state_dict(), optimizer_path)
            training_state["optimizer_state_path"] = "optimizer.pt"
            
        if scheduler is not None:
            # Same approach for scheduler
            scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
            torch.save(scheduler.state_dict(), scheduler_path)
            training_state["scheduler_state_path"] = "scheduler.pt"
            
        # Save epoch and metrics
        # Use TensorJSONEncoder to handle tensors in metrics
        checkpoint_info = {
            "epoch": epoch,
            "metrics": metrics,
            "training_state": training_state
        }
        
        with open(os.path.join(checkpoint_path, "checkpoint_info.json"), "w") as f:
            json.dump(checkpoint_info, f, indent=2, cls=TensorJSONEncoder)
            
        if self.verbose:
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            
        # If this is the best model, update best_path
        if self.save_best_only and self.monitor in metrics:
            self.best_path = checkpoint_path
            
        # Clean up old checkpoints if needed
        if self.max_checkpoints is not None:
            self._cleanup_old_checkpoints()
            
        return checkpoint_path
        
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if there are more than max_checkpoints."""
        # Get all checkpoint directories
        checkpoint_dirs = [d for d in glob.glob(os.path.join(self.output_dir, "checkpoint-*")) if os.path.isdir(d)]
        
        # Sort by epoch number (extracted from directory name)
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
        
        # Keep the best checkpoint if save_best_only is True
        if self.save_best_only and self.best_path:
            checkpoint_dirs = [d for d in checkpoint_dirs if d != self.best_path]
            
        # Remove old checkpoints
        while len(checkpoint_dirs) >= self.max_checkpoints:
            oldest = checkpoint_dirs.pop(0)
            if self.verbose:
                self.logger.info(f"Removing old checkpoint: {oldest}")
            try:
                import shutil
                shutil.rmtree(oldest)
            except Exception as e:
                self.logger.error(f"Error removing checkpoint {oldest}: {e}")
                
    def load(self, checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint.
            device: Device to load the model on.
            
        Returns:
            Dictionary with loaded model, tokenizer, and other info.
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            # Load model and tokenizer
            model = PreTrainedModel.from_pretrained(checkpoint_path)
            tokenizer = PreTrainedTokenizer.from_pretrained(checkpoint_path)
            
            # Move model to device
            model.to(device)
            
            # Load checkpoint info
            checkpoint_info_path = os.path.join(checkpoint_path, "checkpoint_info.json")
            with open(checkpoint_info_path, "r") as f:
                checkpoint_info = json.load(f)
                
            # Load optimizer and scheduler states if they exist
            training_state = checkpoint_info["training_state"]
            if "optimizer_state_path" in training_state:
                optimizer_path = os.path.join(checkpoint_path, training_state["optimizer_state_path"])
                if os.path.exists(optimizer_path):
                    training_state["optimizer_state_dict"] = torch.load(optimizer_path, map_location=device)
                    
            if "scheduler_state_path" in training_state:
                scheduler_path = os.path.join(checkpoint_path, training_state["scheduler_state_path"])
                if os.path.exists(scheduler_path):
                    training_state["scheduler_state_dict"] = torch.load(scheduler_path, map_location=device)
                
            return {
                "model": model,
                "tokenizer": tokenizer,
                "epoch": checkpoint_info["epoch"],
                "metrics": checkpoint_info["metrics"],
                "training_state": training_state
            }
        except Exception as e:
            self.logger.error(f"Error loading checkpoint from {checkpoint_path}: {e}")
            raise
            
    def load_latest(self, device: torch.device) -> Dict[str, Any]:
        """
        Load the latest checkpoint.
        
        Args:
            device: Device to load the model on.
            
        Returns:
            Dictionary with loaded model, tokenizer, and other info.
        """
        # Get all checkpoint directories
        checkpoint_dirs = [d for d in glob.glob(os.path.join(self.output_dir, "checkpoint-*")) if os.path.isdir(d)]
        
        if not checkpoint_dirs:
            raise ValueError(f"No checkpoints found in {self.output_dir}")
            
        # Sort by epoch number (extracted from directory name)
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
        
        # Load the latest checkpoint
        latest_checkpoint = checkpoint_dirs[0]
        return self.load(latest_checkpoint, device)
        
    def load_best(self, device: torch.device) -> Dict[str, Any]:
        """
        Load the best checkpoint based on the monitored metric.
        
        Args:
            device: Device to load the model on.
            
        Returns:
            Dictionary with loaded model, tokenizer, and other info.
        """
        # Get all checkpoint directories
        checkpoint_dirs = [d for d in glob.glob(os.path.join(self.output_dir, "checkpoint-*")) if os.path.isdir(d)]
        
        if not checkpoint_dirs:
            raise ValueError(f"No checkpoints found in {self.output_dir}")
            
        # Find the best checkpoint based on the monitored metric
        best_checkpoint = None
        best_value = float('inf') if self.mode == "min" else float('-inf')
        
        for checkpoint_dir in checkpoint_dirs:
            try:
                # Load checkpoint info
                checkpoint_info_path = os.path.join(checkpoint_dir, "checkpoint_info.json")
                with open(checkpoint_info_path, "r") as f:
                    checkpoint_info = json.load(f)
                    
                # Check if monitored metric is available
                if self.monitor in checkpoint_info["metrics"]:
                    current = checkpoint_info["metrics"][self.monitor]
                    
                    # Update best checkpoint if this one is better
                    if (self.mode == "min" and current < best_value) or (self.mode == "max" and current > best_value):
                        best_value = current
                        best_checkpoint = checkpoint_dir
            except Exception as e:
                self.logger.warning(f"Error reading checkpoint info from {checkpoint_dir}: {e}")
                
        if best_checkpoint is None:
            raise ValueError(f"No checkpoints with metric {self.monitor} found in {self.output_dir}")
            
        return self.load(best_checkpoint, device)
