"""
Early stopping functionality for Gedank Rayze SPLADE Model Trainer.

This module provides early stopping capabilities to prevent overfitting
during model training by monitoring validation metrics.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, Callable, Union, List


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    
    Monitors a specified metric and stops training if no improvement is
    seen for a specified number of epochs.
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 3,
        mode: str = "min",
        verbose: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize early stopping handler.
        
        Args:
            monitor: Metric to monitor (e.g., 'val_loss', 'val_accuracy')
            min_delta: Minimum change in the monitored metric to qualify as improvement
            patience: Number of epochs with no improvement after which training will stop
            mode: One of {'min', 'max'}. 'min' mode: training stops when monitored metric stops decreasing
                  'max' mode: training stops when monitored metric stops increasing
            verbose: Whether to print early stopping information
            logger: Logger to use for messages
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize tracking variables
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.stop_training = False
        
        # Configure message templates
        self.improvement_message = (
            f"Improvement detected: {{current:.6f}} (best: {{best:.6f}}, delta: {{delta:.6f}})"
            if mode == "min" else
            f"Improvement detected: {{current:.6f}} (best: {{best:.6f}}, delta: +{{delta:.6f}})"
        )
        self.no_improvement_message = (
            f"No improvement: {{current:.6f}} (best: {{best:.6f}}, counter: {{counter}}/{{patience}})"
        )
        self.early_stopping_message = (
            f"Early stopping triggered after {{epoch}} epochs. Best {monitor}: {{best:.6f}}"
        )
    
    def __call__(self, epoch: int, metrics: Dict[str, Any]) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics for the current epoch
            
        Returns:
            True if training should stop, False otherwise
        """
        # Ensure the monitored metric exists
        if self.monitor not in metrics:
            self.logger.warning(
                f"Metric '{self.monitor}' not found in metrics. "
                f"Available metrics: {list(metrics.keys())}"
            )
            return False
        
        # Get current value
        current = metrics[self.monitor]
        
        # Check for improvement
        if self.mode == 'min':
            delta = self.best_value - current
            improved = delta > self.min_delta
        else:  # mode == 'max'
            delta = current - self.best_value
            improved = delta > self.min_delta
        
        if improved:
            # Reset counter and update best value
            self.counter = 0
            self.best_value = current
            
            if self.verbose:
                self.logger.info(self.improvement_message.format(
                    current=current, best=self.best_value, delta=delta
                ))
                
            return False
        else:
            # Increment counter
            self.counter += 1
            
            if self.verbose:
                self.logger.info(self.no_improvement_message.format(
                    current=current, best=self.best_value, 
                    counter=self.counter, patience=self.patience
                ))
            
            # Check if patience is exhausted
            if self.counter >= self.patience:
                self.stop_training = True
                if self.verbose:
                    self.logger.info(self.early_stopping_message.format(
                        epoch=epoch, best=self.best_value
                    ))
                return True
            
            return False
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.stop_training = False
        if self.verbose:
            self.logger.info("Early stopping state reset")
    
    def get_best_value(self) -> float:
        """Get the best value seen so far."""
        return self.best_value


class MetricMonitor:
    """
    Monitor multiple metrics during training.
    
    This is a more general version of EarlyStopping that can track multiple metrics
    and execute custom callbacks when specific conditions are met.
    """
    
    def __init__(
        self,
        metrics_to_monitor: List[str],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize metric monitor.
        
        Args:
            metrics_to_monitor: List of metric names to monitor
            logger: Logger to use for messages
        """
        self.metrics_to_monitor = metrics_to_monitor
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize tracking
        self.best_values = {metric: None for metric in metrics_to_monitor}
        self.history = {metric: [] for metric in metrics_to_monitor}
        self.callbacks = []
    
    def add_callback(
        self,
        metric: str,
        condition: Callable[[float, float], bool],
        callback: Callable[[int, str, float, float], None],
        mode: str = "min"
    ) -> None:
        """
        Add a callback to trigger when a condition is met.
        
        Args:
            metric: Metric to monitor
            condition: Function that takes (current_value, best_value) and returns True/False
            callback: Function to call when condition is met
            mode: One of {'min', 'max'} for determining initial best value
        """
        if metric not in self.metrics_to_monitor:
            self.metrics_to_monitor.append(metric)
            self.best_values[metric] = None
            self.history[metric] = []
        
        # Set initial best value based on mode
        if self.best_values[metric] is None:
            self.best_values[metric] = float('inf') if mode == 'min' else float('-inf')
        
        self.callbacks.append({
            'metric': metric,
            'condition': condition,
            'callback': callback,
            'mode': mode
        })
    
    def update(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Update metrics and trigger callbacks if conditions are met.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics for the current epoch
        """
        # Update tracked metrics
        for metric in self.metrics_to_monitor:
            if metric in metrics:
                current_value = metrics[metric]
                
                # Store in history
                self.history[metric].append(current_value)
                
                # Check if this is a new best value
                if self.best_values[metric] is not None:
                    for cb in self.callbacks:
                        if cb['metric'] == metric:
                            if cb['condition'](current_value, self.best_values[metric]):
                                # Update best value if condition is met
                                old_best = self.best_values[metric]
                                if cb['mode'] == 'min':
                                    self.best_values[metric] = min(self.best_values[metric], current_value)
                                else:  # mode == 'max'
                                    self.best_values[metric] = max(self.best_values[metric], current_value)
                                
                                # Call the callback
                                cb['callback'](epoch, metric, current_value, old_best)
    
    def get_best_values(self) -> Dict[str, float]:
        """Get the best values seen for all monitored metrics."""
        return self.best_values
    
    def get_history(self, metric: Optional[str] = None) -> Union[Dict[str, List[float]], List[float]]:
        """
        Get history of metrics.
        
        Args:
            metric: Optional specific metric to get history for
            
        Returns:
            Dictionary of metric histories or list of values for a specific metric
        """
        if metric is not None:
            if metric in self.history:
                return self.history[metric]
            else:
                self.logger.warning(f"Metric '{metric}' not found in history")
                return []
        else:
            return self.history


# Common metric improvement conditions
def is_better_min(current, best, min_delta=0.0):
    """Check if current value is better than best for minimization."""
    return current < (best - min_delta)

def is_better_max(current, best, min_delta=0.0):
    """Check if current value is better than best for maximization."""
    return current > (best + min_delta)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Early stopping example
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=True)
    
    # Simulate 10 epochs with decreasing then stabilizing loss
    metrics = {'val_loss': 0.0}
    for epoch in range(10):
        # Simulate validation loss
        if epoch < 5:
            metrics['val_loss'] = 1.0 - 0.1 * epoch  # Decreasing loss
        else:
            metrics['val_loss'] = 0.5 + 0.01 * (epoch - 5)  # Slightly increasing loss
        
        logger.info(f"Epoch {epoch+1}, val_loss: {metrics['val_loss']:.4f}")
        
        # Check early stopping
        if early_stopping(epoch, metrics):
            logger.info(f"Training would stop at epoch {epoch+1}")
            break
    
    # Metric monitor example
    monitor = MetricMonitor(['val_loss', 'val_accuracy'])
    
    # Add callbacks
    def on_loss_improvement(epoch, metric, current, best):
        logger.info(f"Loss improved at epoch {epoch+1}: {current:.4f} (was {best:.4f})")
    
    def on_accuracy_improvement(epoch, metric, current, best):
        logger.info(f"Accuracy improved at epoch {epoch+1}: {current:.4f} (was {best:.4f})")
    
    monitor.add_callback(
        'val_loss',
        lambda current, best: is_better_min(current, best),
        on_loss_improvement,
        mode='min'
    )
    
    monitor.add_callback(
        'val_accuracy',
        lambda current, best: is_better_max(current, best),
        on_accuracy_improvement,
        mode='max'
    )
    
    # Simulate 10 epochs
    for epoch in range(10):
        # Simulate metrics
        metrics = {
            'val_loss': 1.0 - 0.08 * epoch + 0.005 * epoch**2,  # U-shaped curve
            'val_accuracy': 0.5 + 0.05 * epoch - 0.004 * epoch**2  # Inverted U-shaped curve
        }
        
        logger.info(f"Epoch {epoch+1}, metrics: {metrics}")
        
        # Update monitor
        monitor.update(epoch, metrics)
    
    # Show final best values
    logger.info(f"Best values: {monitor.get_best_values()}")
