"""
Utility modules for Gedank Rayze SPLADE Model Trainer.

Provides common utilities for logging, error handling, and other helper functions.
"""

from src.utils.logging_utils import (
    setup_logging, TrainingLogger, ProgressLogger,
    log_system_info, catch_and_log_exceptions, format_time
)

from src.utils.error_handling import (
    ErrorCategory, ApplicationError, DataError, ModelError, TrainingError,
    EvaluationError, InferenceError, DeviceError, FileSystemError,
    DependencyError, ConfigError, handle_error, try_except, retry,
    validate_file_exists, validate_dir_exists, validate_model_path,
    validate_required_packages, validate_data_format, validate_config
)

# Version
__version__ = "0.1.0"
