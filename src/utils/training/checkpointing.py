"""
Checkpointing functionality for Gedank Rayze SPLADE Model Trainer.

This module provides checkpointing capabilities to save model state during
training and enable recovery from interruptions.
"""

import os
import json
import logging
import time
import shutil
import torch
from typing import Optional, Dict, Any, Callable, Union, List, Tuple
from datetime import datetime


class Checkpointing:
    """
    Model checkpointing handler.
    
    Saves model checkpoints during training and handles loading for recovery.
    """
    
    def __init__(
        self,
        output_dir: str,
        save_best_only: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        verbose: bool = True,
        save_freq: Optional[int] = None,
        max_checkpoints: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize checkpointing handler.
        
        Args:
            output_dir: Directory to save checkpoints
            save_best_only: Only save model when monitored metric improves
            monitor: Metric to monitor (e.g., 'val_loss', 'val_accuracy')
            mode: One of {'min', 'max'}. 'min' mode: checkpoint when monitored metric decreases
                  'max' mode: checkpoint when monitored metric increases
            verbose: Whether to print checkpointing information
            save_freq: Frequency of checkpoints in epochs (if None, only saves based on improvement)
            max_checkpoints: Maximum number of checkpoints to keep (if None, keeps all)
            logger: Logger to use for messages
        """
        self.output_dir = output_dir
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.save_freq = save_freq
        self.max_checkpoints = max_checkpoints
        self.logger = logger or logging.getLogger(__name__)
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = -1
        self.checkpoints = []
        
        # Configure message templates
        self.improvement_message = (
            f"Improvement detected: {{current:.6f}} (best: {{best:.6f}}). Saving checkpoint."
            if mode == "min" else
            f"Improvement detected: {{current:.6f}} (best: {{best:.6f}}). Saving checkpoint."
        )
        self.saving_message = "Saving checkpoint for epoch {epoch} to {path}"
        self.loading_message = "Loading checkpoint from {path}"
        self.no_checkpoint_message = "No checkpoint found at {path}"
        self.removing_message = "Removing old checkpoint: {path}"
    
    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.mode == 'min':
            return current < self.best_value
        else:  # mode == 'max'
            return current > self.best_value
    
    def _get_checkpoint_path(self, epoch: int, metrics: Dict[str, Any]) -> str:
        """Get checkpoint file path for epoch."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metric_value = metrics.get(self.monitor, "NA")
        if isinstance(metric_value, float):
            metric_str = f"{metric_value:.6f}".replace(".", "_")
        else:
            metric_str = str(metric_value)
        
        return os.path.join(
            self.output_dir, 
            f"checkpoint_epoch_{epoch}_{self.monitor}_{metric_str}_{timestamp}.pt"
        )
    
    def _save_model(self, model: torch.nn.Module, tokenizer: Any, epoch: int, 
                   metrics: Dict[str, Any], optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[Any] = None) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model to save
            tokenizer: Tokenizer to save
            epoch: Current epoch number
            metrics: Dictionary of metrics
            optimizer: Optional optimizer to save
            scheduler: Optional scheduler to save
            
        Returns:
            Path to saved checkpoint
        """
        # Get path for checkpoint
        checkpoint_path = self._get_checkpoint_path(epoch, metrics)
        
        # Create checkpoint directory
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model and tokenizer using their native methods
        model.save_pretrained(checkpoint_dir)
        if hasattr(tokenizer, 'save_pretrained'):
            tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            'epoch': epoch,
            'metrics': metrics,
            'best_value': self.best_value,
            'best_epoch': self.best_epoch
        }
        
        # Add optimizer and scheduler if provided
        if optimizer is not None:
            training_state['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            training_state['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save training state
        torch.save(training_state, checkpoint_path)
        
        # Save checkpoint info
        checkpoint_info = {
            'epoch': epoch,
            'path': checkpoint_path,
            'metrics': metrics,
            'timestamp': time.time()
        }
        self.checkpoints.append(checkpoint_info)
        
        # Log
        if self.verbose:
            self.logger.info(self.saving_message.format(epoch=epoch, path=checkpoint_path))
        
        return checkpoint_path
    
    def _manage_checkpoints(self) -> None:
        """Manage number of checkpoints."""
        if self.max_checkpoints is not None and len(self.checkpoints) > self.max_checkpoints:
            # Sort by timestamp (oldest first)
            self.checkpoints.sort(key=lambda x: x['timestamp'])
            
            # Keep the best checkpoint regardless of age
            best_checkpoint = next(
                (cp for cp in self.checkpoints if cp['epoch'] == self.best_epoch), 
                None
            )
            
            # Remove oldest checkpoints, but keep the best
            while len(self.checkpoints) > self.max_checkpoints:
                to_remove = self.checkpoints[0]
                
                # Don't remove the best checkpoint
                if best_checkpoint is not None and to_remove['path'] == best_checkpoint['path']:
                    # Try the next oldest
                    if len(self.checkpoints) > 1:
                        to_remove = self.checkpoints[1]
                        del self.checkpoints[1]
                    else:
                        # No more checkpoints to remove
                        break
                else:
                    del self.checkpoints[0]
                
                # Remove checkpoint file
                checkpoint_path = to_remove['path']
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    if self.verbose:
                        self.logger.info(self.removing_message.format(path=checkpoint_path))
    
    def save(self, model: torch.nn.Module, tokenizer: Any, epoch: int, 
             metrics: Dict[str, Any], optimizer: Optional[torch.optim.Optimizer] = None,
             scheduler: Optional[Any] = None) -> Optional[str]:
        """
        Save model checkpoint if conditions are met.
        
        Args:
            model: PyTorch model to save
            tokenizer: Tokenizer to save
            epoch: Current epoch number
            metrics: Dictionary of metrics
            optimizer: Optional optimizer to save
            scheduler: Optional scheduler to save
            
        Returns:
            Path to saved checkpoint if one was saved, None otherwise
        """
        # Check if we should save based on frequency
        save_based_on_freq = (self.save_freq is not None and epoch % self.save_freq == 0)
        
        # Check if we should save based on improvement
        save_based_on_improvement = False
        current_value = metrics.get(self.monitor)
        
        if current_value is not None:
            if self._is_improvement(current_value):
                # Update best value
                self.best_value = current_value
                self.best_epoch = epoch
                
                if self.verbose:
                    self.logger.info(self.improvement_message.format(
                        current=current_value, best=self.best_value
                    ))
                
                save_based_on_improvement = True
        
        # Determine if we should save
        should_save = save_based_on_freq
        if self.save_best_only:
            should_save = should_save or save_based_on_improvement
        else:
            should_save = True
        
        # Save if conditions are met
        if should_save:
            checkpoint_path = self._save_model(
                model, tokenizer, epoch, metrics, optimizer, scheduler
            )
            
            # Manage number of checkpoints
            self._manage_checkpoints()
            
            return checkpoint_path
        
        return None
    
    def load_latest(self, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Load the latest checkpoint.
        
        Args:
            device: Device to load model to
            
        Returns:
            Dictionary with loaded items (model, tokenizer, etc.)
            
        Raises:
            FileNotFoundError: If no checkpoint is found
        """
        # Find checkpoints in directory
        checkpoint_files = [
            f for f in os.listdir(self.output_dir) 
            if f.startswith("checkpoint_epoch_") and f.endswith(".pt")
        ]
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in {self.output_dir}")
        
        # Sort by timestamp (newest first)
        checkpoint_files.sort(reverse=True)
        latest_checkpoint = os.path.join(self.output_dir, checkpoint_files[0])
        
        return self.load(latest_checkpoint, device)
    
    def load_best(self, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Load the best checkpoint based on monitored metric.
        
        Args:
            device: Device to load model to
            
        Returns:
            Dictionary with loaded items (model, tokenizer, etc.)
            
        Raises:
            FileNotFoundError: If no checkpoint is found
        """
        # Find checkpoints in directory
        checkpoint_files = [
            f for f in os.listdir(self.output_dir) 
            if f.startswith("checkpoint_epoch_") and f.endswith(".pt")
        ]
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in {self.output_dir}")
        
        # Parse metrics from filenames
        best_checkpoint = None
        best_value = float('inf') if self.mode == 'min' else float('-inf')
        
        for filename in checkpoint_files:
            # Try to extract metric value
            try:
                parts = filename.split('_')
                monitor_idx = parts.index(self.monitor)
                metric_str = parts[monitor_idx + 1].replace("_", ".")
                metric_value = float(metric_str)
                
                # Check if this is the best
                if (self.mode == 'min' and metric_value < best_value) or \
                   (self.mode == 'max' and metric_value > best_value):
                    best_value = metric_value
                    best_checkpoint = filename
            except (ValueError, IndexError):
                # Skip if we can't parse the metric
                continue
        
        if best_checkpoint is None:
            # Fallback to the latest
            return self.load_latest(device)
        
        best_checkpoint_path = os.path.join(self.output_dir, best_checkpoint)
        return self.load(best_checkpoint_path, device)
    
    def load(self, checkpoint_path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Load a specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model to
            
        Returns:
            Dictionary with loaded items (model, tokenizer, etc.)
            
        Raises:
            FileNotFoundError: If checkpoint file is not found
        """
        if not os.path.exists(checkpoint_path):
            error_msg = self.no_checkpoint_message.format(path=checkpoint_path)
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if self.verbose:
            self.logger.info(self.loading_message.format(path=checkpoint_path))
        
        # Load checkpoint directory
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        # Import here to avoid circular imports
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        
        # Load model and tokenizer
        try:
            model = AutoModelForMaskedLM.from_pretrained(checkpoint_dir)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            
            # Move to device if specified
            if device is not None:
                model = model.to(device)
            
            # Load training state
            training_state = torch.load(
                checkpoint_path, 
                map_location=device if device is not None else torch.device('cpu')
            )
            
            # Extract state
            epoch = training_state['epoch']
            metrics = training_state['metrics']
            
            # Create result dictionary
            result = {
                'model': model,
                'tokenizer': tokenizer,
                'epoch': epoch,
                'metrics': metrics,
                'training_state': training_state
            }
            
            # Update tracking variables
            self.best_value = training_state.get('best_value', self.best_value)
            self.best_epoch = training_state.get('best_epoch', self.best_epoch)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def resume_training(
        self, 
        checkpoint_path: Optional[str] = None, 
        device: Optional[torch.device] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Tuple[torch.nn.Module, Any, int, Dict[str, Any]]:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file (if None, load latest)
            device: Device to load model to
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            
        Returns:
            Tuple of (model, tokenizer, epoch, metrics)
        """
        # Load checkpoint
        if checkpoint_path is not None:
            checkpoint = self.load(checkpoint_path, device)
        else:
            # Try loading the latest checkpoint
            try:
                checkpoint = self.load_latest(device)
            except FileNotFoundError:
                self.logger.warning("No checkpoint found to resume from.")
                raise
        
        # Extract components
        model = checkpoint['model']
        tokenizer = checkpoint['tokenizer']
        epoch = checkpoint['epoch']
        metrics = checkpoint['metrics']
        training_state = checkpoint['training_state']
        
        # Restore optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in training_state:
            optimizer.load_state_dict(training_state['optimizer_state_dict'])
        
        # Restore scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in training_state:
            scheduler.load_state_dict(training_state['scheduler_state_dict'])
        
        return model, tokenizer, epoch, metrics


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Example checkpointing
    checkpointer = Checkpointing(
        output_dir="./checkpoints",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=True,
        save_freq=2,  # Save every 2 epochs
        max_checkpoints=3  # Keep only the 3 most recent checkpoints
    )
    
    # Create dummy model and tokenizer
    import torch.nn as nn
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.fc(x)
        
        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)
            torch.save(self.state_dict(), os.path.join(path, "model.pt"))
    
    class DummyTokenizer:
        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "tokenizer.json"), "w") as f:
                json.dump({"vocab_size": 1000}, f)
    
    # Create model and tokenizer
    model = DummyModel()
    tokenizer = DummyTokenizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Simulate training and checkpointing
    for epoch in range(10):
        # Simulate metrics
        metrics = {
            "train_loss": 1.0 - 0.08 * epoch + 0.005 * epoch**2,  # U-shaped curve
            "val_loss": 0.8 - 0.07 * epoch + 0.006 * epoch**2,  # U-shaped curve
            "val_accuracy": 0.5 + 0.05 * epoch - 0.004 * epoch**2  # Inverted U-shaped curve
        }
        
        logger.info(f"Epoch {epoch+1}, metrics: {metrics}")
        
        # Save checkpoint
        checkpoint_path = checkpointer.save(
            model=model,
            tokenizer=tokenizer,
            epoch=epoch+1,
            metrics=metrics,
            optimizer=optimizer
        )
        
        if checkpoint_path:
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Demonstrate checkpoint loading
    try:
        loaded = checkpointer.load_best()
        logger.info(f"Loaded best checkpoint from epoch {loaded['epoch']}")
        logger.info(f"Best metrics: {loaded['metrics']}")
    except FileNotFoundError as e:
        logger.error(f"Error loading checkpoint: {e}")
