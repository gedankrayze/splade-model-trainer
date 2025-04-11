#!/usr/bin/env python3
"""
Error handling utilities for Gedank Rayze SPLADE Model Trainer.

This module provides enhanced error handling with detailed error messages,
automatic recovery mechanisms, and error classification.
"""

import functools
import inspect
import logging
import os
import sys
import traceback
from enum import Enum, auto
from typing import Optional, Callable, Any, Dict, List, Type, Union, Tuple


class ErrorCategory(Enum):
    """Categories of errors that can occur in the application."""
    
    DATA_ERROR = auto()           # Data loading, parsing, or processing errors
    MODEL_ERROR = auto()          # Model-related errors (loading, initialization)
    TRAINING_ERROR = auto()       # Errors during training process
    EVALUATION_ERROR = auto()     # Errors during model evaluation
    INFERENCE_ERROR = auto()      # Errors during inference/prediction
    DEVICE_ERROR = auto()         # Device-related errors (GPU, memory)
    FILE_SYSTEM_ERROR = auto()    # File I/O errors
    DEPENDENCY_ERROR = auto()     # Missing or incompatible dependencies
    CONFIG_ERROR = auto()         # Configuration-related errors
    UNKNOWN_ERROR = auto()        # Unknown or uncategorized errors


class ApplicationError(Exception):
    """
    Base exception class for application-specific errors.
    Provides additional context and categorization for errors.
    """
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
        original_exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize application error.
        
        Args:
            message: Error message
            category: Error category
            original_exception: Original exception that caused this error
            context: Additional context information
        """
        self.message = message
        self.category = category
        self.original_exception = original_exception
        self.context = context or {}
        
        # Build full message
        full_message = f"{message}"
        if original_exception:
            full_message += f" (caused by: {type(original_exception).__name__}: {str(original_exception)})"
        
        super().__init__(full_message)
    
    def __str__(self):
        """String representation of the error."""
        base_str = super().__str__()
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base_str} [context: {context_str}]"
        return base_str
    
    def get_traceback(self) -> str:
        """Get formatted traceback from original exception."""
        if self.original_exception:
            return "".join(traceback.format_exception(
                type(self.original_exception),
                self.original_exception,
                self.original_exception.__traceback__
            ))
        return ""
    
    def get_full_report(self) -> str:
        """Get detailed error report with all available information."""
        lines = [
            f"Error: {self.message}",
            f"Category: {self.category.name}",
        ]
        
        if self.context:
            lines.append("Context:")
            for key, value in self.context.items():
                lines.append(f"  {key}: {value}")
        
        if self.original_exception:
            lines.append(f"Original exception: {type(self.original_exception).__name__}: {str(self.original_exception)}")
            lines.append("Traceback:")
            lines.append(self.get_traceback())
        
        return "\n".join(lines)


# Specific error classes for common error categories

class DataError(ApplicationError):
    """Data-related errors."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None, **context):
        super().__init__(message, ErrorCategory.DATA_ERROR, original_exception, context)


class ModelError(ApplicationError):
    """Model-related errors."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None, **context):
        super().__init__(message, ErrorCategory.MODEL_ERROR, original_exception, context)


class TrainingError(ApplicationError):
    """Training-related errors."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None, **context):
        super().__init__(message, ErrorCategory.TRAINING_ERROR, original_exception, context)


class EvaluationError(ApplicationError):
    """Evaluation-related errors."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None, **context):
        super().__init__(message, ErrorCategory.EVALUATION_ERROR, original_exception, context)


class InferenceError(ApplicationError):
    """Inference-related errors."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None, **context):
        super().__init__(message, ErrorCategory.INFERENCE_ERROR, original_exception, context)


class DeviceError(ApplicationError):
    """Device-related errors."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None, **context):
        super().__init__(message, ErrorCategory.DEVICE_ERROR, original_exception, context)


class FileSystemError(ApplicationError):
    """File system-related errors."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None, **context):
        super().__init__(message, ErrorCategory.FILE_SYSTEM_ERROR, original_exception, context)


class DependencyError(ApplicationError):
    """Dependency-related errors."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None, **context):
        super().__init__(message, ErrorCategory.DEPENDENCY_ERROR, original_exception, context)


class ConfigError(ApplicationError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None, **context):
        super().__init__(message, ErrorCategory.CONFIG_ERROR, original_exception, context)


# Error handling utilities

def handle_error(
    error: Exception,
    logger: logging.Logger,
    exit_on_error: bool = False,
    log_level: int = logging.ERROR,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Handle an error by logging it and optionally exiting.
    
    Args:
        error: The exception to handle
        logger: Logger to use
        exit_on_error: Whether to exit the application
        log_level: Logging level to use
        context: Additional context information
    """
    # Convert to ApplicationError if it's not already
    if not isinstance(error, ApplicationError):
        error = ApplicationError(
            str(error),
            ErrorCategory.UNKNOWN_ERROR,
            error,
            context
        )
    
    # Log the error
    logger.log(
        log_level,
        f"{error.category.name}: {error}",
        exc_info=error.original_exception
    )
    
    # Print full report for critical errors
    if log_level >= logging.ERROR:
        logger.debug(error.get_full_report())
    
    # Exit if requested
    if exit_on_error:
        logger.critical("Exiting due to unrecoverable error")
        sys.exit(1)


def try_except(
    error_type: Type[ApplicationError] = ApplicationError,
    message: str = "An error occurred",
    logger: Optional[logging.Logger] = None,
    exit_on_error: bool = False,
    default_return: Any = None
) -> Callable:
    """
    Decorator to handle exceptions and convert them to appropriate application errors.
    
    Args:
        error_type: Type of application error to create
        message: Error message
        logger: Logger to use
        exit_on_error: Whether to exit on error
        default_return: Default value to return on error
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get function call context
                context = {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                
                # Create application error
                app_error = error_type(message, e, **context)
                
                # Handle error
                handle_error(app_error, logger, exit_on_error)
                
                # Return default value
                return default_return
        return wrapper
    return decorator


def validate_file_exists(file_path: str, error_message: Optional[str] = None) -> None:
    """
    Validate that a file exists, raising FileSystemError if not.
    
    Args:
        file_path: Path to file
        error_message: Custom error message
    
    Raises:
        FileSystemError: If file does not exist
    """
    if not os.path.exists(file_path):
        message = error_message or f"File not found: {file_path}"
        raise FileSystemError(message, context={'file_path': file_path})
    
    if not os.path.isfile(file_path):
        message = error_message or f"Path exists but is not a file: {file_path}"
        raise FileSystemError(message, context={'file_path': file_path})


def validate_dir_exists(dir_path: str, error_message: Optional[str] = None, create: bool = False) -> None:
    """
    Validate that a directory exists, raising FileSystemError if not.
    
    Args:
        dir_path: Path to directory
        error_message: Custom error message
        create: Whether to create directory if it doesn't exist
    
    Raises:
        FileSystemError: If directory does not exist and create is False
    """
    if not os.path.exists(dir_path):
        if create:
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                message = error_message or f"Failed to create directory: {dir_path}"
                raise FileSystemError(message, e, dir_path=dir_path)
        else:
            message = error_message or f"Directory not found: {dir_path}"
            raise FileSystemError(message, context={'dir_path': dir_path})
    
    if not os.path.isdir(dir_path):
        message = error_message or f"Path exists but is not a directory: {dir_path}"
        raise FileSystemError(message, context={'dir_path': dir_path})


def validate_model_path(model_path: str, error_message: Optional[str] = None) -> None:
    """
    Validate that a model path exists and contains required model files.
    
    Args:
        model_path: Path to model directory
        error_message: Custom error message
    
    Raises:
        ModelError: If model path is invalid
    """
    # Check that directory exists
    try:
        validate_dir_exists(model_path)
    except FileSystemError as e:
        message = error_message or f"Invalid model path: {model_path}"
        raise ModelError(message, e, model_path=model_path)
    
    # Check for required files (PyTorch or Transformers)
    has_pytorch = any(f.endswith('.pt') or f.endswith('.pth') for f in os.listdir(model_path))
    has_transformers = os.path.exists(os.path.join(model_path, 'config.json'))
    
    if not (has_pytorch or has_transformers):
        message = error_message or f"No model files found in: {model_path}"
        raise ModelError(message, context={'model_path': model_path})


def validate_required_packages(
    package_names: List[str],
    error_message: Optional[str] = None,
    min_versions: Optional[Dict[str, str]] = None
) -> None:
    """
    Validate that required packages are installed.
    
    Args:
        package_names: List of required package names
        error_message: Custom error message
        min_versions: Dictionary mapping package names to minimum versions
    
    Raises:
        DependencyError: If a required package is missing or version is too low
    """
    import importlib
    import pkg_resources
    
    min_versions = min_versions or {}
    missing_packages = []
    outdated_packages = []
    
    for package_name in package_names:
        try:
            pkg = importlib.import_module(package_name)
            
            # Check version if specified
            if package_name in min_versions:
                min_version = min_versions[package_name]
                try:
                    installed_version = pkg_resources.get_distribution(package_name).version
                    if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                        outdated_packages.append((package_name, installed_version, min_version))
                except Exception:
                    # Can't determine version, assume it's ok
                    pass
                
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages or outdated_packages:
        context = {}
        message_parts = []
        
        if missing_packages:
            context['missing_packages'] = missing_packages
            packages_str = ', '.join(missing_packages)
            message_parts.append(f"Missing packages: {packages_str}")
        
        if outdated_packages:
            context['outdated_packages'] = outdated_packages
            outdated_str = ', '.join(f"{pkg} (found {found}, need {required})" 
                                    for pkg, found, required in outdated_packages)
            message_parts.append(f"Outdated packages: {outdated_str}")
        
        message = error_message or '; '.join(message_parts)
        raise DependencyError(message, context=context)


def validate_data_format(
    data: Any,
    validation_func: Callable[[Any], bool],
    error_message: Optional[str] = None
) -> None:
    """
    Validate data format using a validation function.
    
    Args:
        data: Data to validate
        validation_func: Function that returns True if data is valid
        error_message: Custom error message
    
    Raises:
        DataError: If data validation fails
    """
    if not validation_func(data):
        message = error_message or "Invalid data format"
        raise DataError(message, context={'data_sample': str(data)[:100]})


def validate_config(
    config: Dict[str, Any],
    required_keys: List[str],
    optional_keys: Optional[List[str]] = None,
    key_validators: Optional[Dict[str, Callable[[Any], bool]]] = None,
    error_message: Optional[str] = None
) -> None:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        optional_keys: List of optional keys
        key_validators: Dictionary mapping keys to validation functions
        error_message: Custom error message
    
    Raises:
        ConfigError: If configuration is invalid
    """
    optional_keys = optional_keys or []
    key_validators = key_validators or {}
    
    # Check for missing required keys
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        message = error_message or f"Missing required configuration keys: {', '.join(missing_keys)}"
        raise ConfigError(message, context={'missing_keys': missing_keys})
    
    # Check for unknown keys
    allowed_keys = required_keys + optional_keys
    unknown_keys = [key for key in config if key not in allowed_keys]
    if unknown_keys:
        # Log warning but don't raise error
        logger = logging.getLogger(__name__)
        logger.warning(f"Unknown configuration keys: {', '.join(unknown_keys)}")
    
    # Validate values
    invalid_keys = []
    for key, validator in key_validators.items():
        if key in config and not validator(config[key]):
            invalid_keys.append(key)
    
    if invalid_keys:
        message = error_message or f"Invalid configuration values for keys: {', '.join(invalid_keys)}"
        raise ConfigError(message, context={'invalid_keys': invalid_keys})


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Optional[Union[Type[Exception], Tuple[Type[Exception], ...]]] = None,
    logger: Optional[logging.Logger] = None
) -> Callable:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts (seconds)
        backoff_factor: Backoff factor for delay
        exceptions: Exception types to catch and retry
        logger: Logger to use
        
    Returns:
        Decorated function
    """
    exceptions = exceptions or Exception
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts: {e}")
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    
                    # Wait before retrying
                    import time
                    time.sleep(current_delay)
                    
                    # Increase delay for next attempt
                    current_delay *= backoff_factor
                    attempt += 1
        
        return wrapper
    
    return decorator


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    # Using application errors
    try:
        raise DataError("Failed to load data file", context={'file': 'data.json'})
    except ApplicationError as e:
        handle_error(e, logger)
        print(e.get_full_report())
    
    # Using decorators
    @try_except(DataError, "Error in data processing", logger)
    def process_data():
        raise ValueError("Invalid value")
    
    process_data()  # Will catch and log the error
    
    # Using validation functions
    try:
        validate_file_exists("non_existent_file.txt")
    except FileSystemError as e:
        handle_error(e, logger)
    
    # Using retry
    @retry(max_attempts=3, delay=0.1, logger=logger)
    def unreliable_function():
        import random
        if random.random() < 0.8:
            raise ConnectionError("Simulated connection error")
        return "Success"
    
    try:
        result = unreliable_function()
        print(f"Result: {result}")
    except ConnectionError:
        print("Failed all retry attempts")
