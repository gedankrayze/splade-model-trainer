#!/usr/bin/env python3
"""
Logging utilities for Gedank Rayze SPLADE Model Trainer.

This module provides enhanced logging capabilities with configurable verbosity,
structured formatting, and file output options.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List


class CustomFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    # Color codes
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[1;91m',  # Bold Red
        'RESET': '\033[0m'  # Reset
    }

    def __init__(self, use_colors=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_colors = use_colors and sys.stdout.isatty()  # Only use colors for TTY devices

    def format(self, record):
        log_message = super().format(record)
        levelname = record.levelname
        
        if self.use_colors and levelname in self.COLORS:
            return f"{self.COLORS[levelname]}{log_message}{self.COLORS['RESET']}"
        return log_message


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra attributes
        if hasattr(record, 'props'):
            log_data.update(record.props)
        
        return json.dumps(log_data)


class ProgressLogger:
    """
    Logger that handles progress tracking and reporting.
    Can be used as a drop-in replacement for tqdm in certain contexts.
    """

    def __init__(self, logger, total, desc="Progress", logging_interval=5):
        """
        Initialize progress logger.
        
        Args:
            logger: Logger instance
            total: Total number of items
            desc: Description for progress
            logging_interval: How often to log progress (in seconds)
        """
        self.logger = logger
        self.total = total
        self.desc = desc
        self.n = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.logging_interval = logging_interval
        
        # Initial log
        self.logger.info(f"Starting {self.desc}: 0/{self.total} (0.0%)")
    
    def update(self, n=1):
        """
        Update progress.
        
        Args:
            n: Number of items to increment
        """
        self.n += n
        current_time = time.time()
        
        # Log if enough time has passed since last log
        if current_time - self.last_log_time >= self.logging_interval:
            self._log_progress()
            self.last_log_time = current_time
    
    def _log_progress(self):
        """Log current progress."""
        progress_pct = (self.n / self.total) * 100 if self.total > 0 else 0
        elapsed = time.time() - self.start_time
        
        # Calculate items per second and ETA
        items_per_sec = self.n / elapsed if elapsed > 0 else 0
        eta = (self.total - self.n) / items_per_sec if items_per_sec > 0 else float('inf')
        
        # Format ETA
        if eta == float('inf'):
            eta_str = "unknown"
        elif eta > 3600:
            eta_str = f"{eta / 3600:.1f}h"
        elif eta > 60:
            eta_str = f"{eta / 60:.1f}m"
        else:
            eta_str = f"{eta:.1f}s"
        
        self.logger.info(
            f"{self.desc}: {self.n}/{self.total} ({progress_pct:.1f}%) "
            f"[{items_per_sec:.1f} it/s, ETA: {eta_str}]"
        )
    
    def close(self):
        """Complete the progress tracking and log final statistics."""
        elapsed = time.time() - self.start_time
        items_per_sec = self.n / elapsed if elapsed > 0 else 0
        
        self.logger.info(
            f"Completed {self.desc}: {self.n}/{self.total} (100.0%) "
            f"in {elapsed:.1f}s [{items_per_sec:.1f} it/s]"
        )


def setup_logging(
    log_dir: Optional[str] = None,
    level: str = "INFO",
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    json_output: bool = False,
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_filename: Optional[str] = None,
    use_colors: bool = True,
    capture_warnings: bool = True
) -> logging.Logger:
    """
    Set up logging with enhanced features.
    
    Args:
        log_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Format string for logs
        json_output: Whether to output logs in JSON format
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_filename: Custom log filename (defaults to splade_trainer_{timestamp}.log)
        use_colors: Whether to use colors in console output
        capture_warnings: Whether to capture warnings from warnings module
        
    Returns:
        Configured root logger
    """
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Capture warnings if requested
    if capture_warnings:
        logging.captureWarnings(True)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        
        if json_output:
            console_formatter = JSONFormatter()
        else:
            console_formatter = CustomFormatter(use_colors=use_colors, fmt=format_string)
            
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_to_file and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
        if not log_filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"splade_trainer_{timestamp}.log"
        
        log_path = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(log_path)
        
        if json_output:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(format_string)
            
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Log the file location
        root_logger.info(f"Logging to file: {log_path}")
    
    return root_logger


class TrainingLogger:
    """
    Specialized logger for training processes with metrics tracking.
    """
    
    def __init__(self, name, log_dir=None, extra=None):
        """
        Initialize training logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            extra: Extra properties to add to all log records
        """
        self.name = name
        self.log_dir = log_dir
        
        # Set up logger
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.metrics = {}
        self.start_time = None
        self.end_time = None
    
    def debug(self, msg, *args, **kwargs):
        """Log a debug message."""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        """Log an info message."""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """Log a warning message."""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """Log an error message."""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        """Log a critical message."""
        self.logger.critical(msg, *args, **kwargs)
        
    def start_training(self, config=None):
        """
        Log training start.
        
        Args:
            config: Training configuration
        """
        self.start_time = time.time()
        self.info("Training started")
        
        if config:
            self.info(f"Training configuration: {json.dumps(config, indent=2)}")
    
    def log_epoch(self, epoch, metrics):
        """
        Log epoch metrics.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        self.info(f"Epoch {epoch} completed")
        self.info(f"Metrics: {json.dumps(metrics, indent=2)}")
        
        # Store metrics
        self.metrics[epoch] = metrics
    
    def log_evaluation(self, eval_name, metrics):
        """
        Log evaluation metrics.
        
        Args:
            eval_name: Evaluation name
            metrics: Dictionary of metrics
        """
        self.info(f"Evaluation '{eval_name}' completed")
        self.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    def log_error(self, error_msg, exc_info=None):
        """
        Log error with optional exception info.
        
        Args:
            error_msg: Error message
            exc_info: Exception info
        """
        self.logger.error(error_msg, exc_info=exc_info)
    
    def end_training(self, final_metrics=None):
        """
        Log training end.
        
        Args:
            final_metrics: Final metrics
        """
        self.end_time = time.time()
        duration = self.end_time - self.start_time if self.start_time else 0
        
        log_data = {
            'duration_seconds': duration,
            'duration_formatted': format_time(duration)
        }
        
        if final_metrics:
            log_data['final_metrics'] = final_metrics
        
        self.info(f"Training completed in {format_time(duration)}")
        if final_metrics:
            self.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
    
    def save_metrics(self, filename='metrics.json'):
        """
        Save training metrics to file.
        
        Args:
            filename: Filename for metrics
        """
        if not self.log_dir:
            self.logger.warning("Cannot save metrics: log_dir not set")
            return
            
        filepath = os.path.join(self.log_dir, filename)
        
        training_summary = {
            'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            'duration_seconds': self.end_time - self.start_time if self.start_time and self.end_time else None,
            'metrics_by_epoch': self.metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(training_summary, f, indent=2)
            
        self.info(f"Training metrics saved to {filepath}")


def format_time(seconds):
    """
    Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{minutes}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"


def log_system_info():
    """Log system information for debugging purposes."""
    import platform
    try:
        import psutil
    except ImportError:
        psutil = None
    
    try:
        import torch
    except ImportError:
        torch = None
    
    system_info = {
        'python_version': platform.python_version(),
        'os': platform.platform(),
        'cpu': platform.processor(),
    }
    
    if psutil:
        system_info.update({
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2)
        })
    
    if torch:
        system_info.update({
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        })
    
    logger = logging.getLogger(__name__)
    logger.info(f"System information: {json.dumps(system_info, indent=2)}")
    return system_info


def catch_and_log_exceptions(logger=None):
    """
    Decorator to catch and log exceptions from functions.
    
    Args:
        logger: Logger to use (defaults to function's module logger)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
                
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Exception in {func.__name__}: {str(e)}",
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    log_dir = "logs"
    logger = setup_logging(log_dir=log_dir, level="DEBUG")
    
    # Log some messages
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Example with adaptation
    training_logger = TrainingLogger("training_example", log_dir=log_dir)
    
    # Configure and start training
    config = {
        'model': 'splade',
        'learning_rate': 5e-5,
        'batch_size': 16,
        'epochs': 3
    }
    training_logger.start_training(config)
    
    # Log epochs
    for epoch in range(1, 4):
        # Simulate training
        time.sleep(1)
        
        metrics = {
            'loss': 1.0 / (epoch + 1),
            'accuracy': 0.5 + epoch * 0.1
        }
        training_logger.log_epoch(epoch, metrics)
    
    # End training
    training_logger.end_training({'final_loss': 0.2, 'final_accuracy': 0.85})
    
    # Save metrics
    training_logger.save_metrics()
    
    # Log system info
    log_system_info()
