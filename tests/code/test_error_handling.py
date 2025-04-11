#!/usr/bin/env python3
"""
Test script for error handling and logging system.

This script demonstrates the features of the error handling and logging system
by simulating various error scenarios and showing how they are handled.
"""

import os
import sys
import time
import argparse
import random
from typing import Any, Dict, List, Optional

# Add the project root to the path so we can import the src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils import (
    setup_logging, TrainingLogger, ProgressLogger,
    log_system_info, catch_and_log_exceptions,
    DataError, ModelError, TrainingError, DeviceError, FileSystemError,
    DependencyError, ConfigError, handle_error, try_except, retry,
    validate_file_exists, validate_dir_exists, validate_model_path,
    validate_required_packages, validate_data_format, validate_config
)


def setup_test_environment():
    """Set up the test environment with a logger."""
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = setup_logging(
        log_dir=log_dir,
        level="DEBUG",
        use_colors=True
    )
    
    return logger, log_dir


def test_basic_logging(logger):
    """Test basic logging functionality."""
    logger.info("=== Testing Basic Logging ===")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")


def test_progress_logger(logger, log_dir):
    """Test the progress logger."""
    logger.info("=== Testing Progress Logger ===")
    
    # Create a progress logger
    progress = ProgressLogger(logger, total=100, desc="Processing items")
    
    # Simulate progress
    for i in range(100):
        # Simulate processing
        time.sleep(0.01)
        progress.update()
        
        # Simulate occasional errors
        if i % 25 == 0 and i > 0:
            logger.warning(f"Milestone reached: {i}/100")
    
    progress.close()


def test_training_logger(logger, log_dir):
    """Test the training logger."""
    logger.info("=== Testing Training Logger ===")
    
    # Create a training logger
    training_logger = TrainingLogger("test_training", log_dir)
    
    # Start training
    config = {
        "model": "test_model",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 3
    }
    training_logger.start_training(config)
    
    # Log epochs
    for epoch in range(1, 4):
        time.sleep(0.5)  # Simulate training
        
        # Generate random metrics
        metrics = {
            "loss": 1.0 / (epoch + 1),
            "accuracy": 0.5 + epoch * 0.1,
            "examples_per_second": 100 + epoch * 20
        }
        
        training_logger.log_epoch(epoch, metrics)
    
    # End training
    training_logger.end_training({
        "final_loss": 0.25,
        "final_accuracy": 0.85
    })
    
    # Save metrics
    training_logger.save_metrics()


def test_exception_handling(logger):
    """Test exception handling."""
    logger.info("=== Testing Exception Handling ===")
    
    # 1. Basic error handling
    try:
        logger.info("Simulating a basic error...")
        raise ValueError("This is a test error")
    except Exception as e:
        logger.error(f"Caught error: {e}")
    
    # 2. Application-specific errors
    try:
        logger.info("Simulating a data error...")
        raise DataError("Failed to parse data file", context={"file": "data.json"})
    except Exception as e:
        logger.error(f"Caught application error: {e}")
        logger.info(f"Error category: {e.category.name}")
    
    # 3. Nested exceptions
    try:
        logger.info("Simulating a nested error...")
        try:
            # Inner exception
            raise ValueError("Database connection failed")
        except ValueError as inner_e:
            # Wrap in an application error
            raise ModelError("Model initialization failed", inner_e, model="test_model")
    except Exception as e:
        logger.error(f"Caught nested error: {e}")
        if hasattr(e, 'original_exception'):
            logger.info(f"Original exception: {e.original_exception}")


@try_except(DataError, "Error in process_data function")
def process_data(data):
    """Process data with automatic error handling."""
    if not isinstance(data, list):
        raise TypeError(f"Expected list, got {type(data).__name__}")
    
    if not data:
        raise ValueError("Empty data list")
        
    return [item * 2 for item in data]


@retry(max_attempts=3, delay=0.1)
def unreliable_function():
    """A function that fails sometimes but will eventually succeed."""
    if random.random() < 0.7:
        raise ConnectionError("Simulated connection error")
    return "Success!"


@catch_and_log_exceptions()
def function_with_logged_exceptions():
    """A function with automatic exception logging."""
    raise RuntimeError("This exception will be automatically logged")


def test_decorators(logger):
    """Test the decorator utilities."""
    logger.info("=== Testing Decorators ===")
    
    # 1. try_except decorator
    logger.info("Testing try_except decorator...")
    
    # Should succeed
    result = process_data([1, 2, 3])
    logger.info(f"process_data result: {result}")
    
    # Should handle error
    result = process_data("not a list")
    logger.info(f"process_data with error result: {result}")
    
    # 2. retry decorator
    logger.info("Testing retry decorator...")
    try:
        result = unreliable_function()
        logger.info(f"unreliable_function result: {result}")
    except Exception as e:
        logger.error(f"Retry finally failed: {e}")
    
    # 3. catch_and_log_exceptions decorator
    logger.info("Testing catch_and_log_exceptions decorator...")
    try:
        function_with_logged_exceptions()
    except Exception as e:
        logger.info(f"Exception was caught outside: {e}")


def test_validation_functions(logger):
    """Test the validation utilities."""
    logger.info("=== Testing Validation Functions ===")
    
    # 1. File validation
    logger.info("Testing file validation...")
    
    existing_file = __file__  # This script exists
    non_existent_file = "this_file_does_not_exist.txt"
    
    try:
        validate_file_exists(existing_file)
        logger.info(f"Validation passed for existing file: {existing_file}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    try:
        validate_file_exists(non_existent_file)
    except Exception as e:
        logger.info(f"Expected error caught: {e}")
    
    # 2. Directory validation
    logger.info("Testing directory validation...")
    
    existing_dir = os.path.dirname(__file__)
    non_existent_dir = "this_directory_does_not_exist"
    
    try:
        validate_dir_exists(existing_dir)
        logger.info(f"Validation passed for existing directory: {existing_dir}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    try:
        validate_dir_exists(non_existent_dir)
    except Exception as e:
        logger.info(f"Expected error caught: {e}")
    
    try:
        validate_dir_exists(non_existent_dir, create=True)
        logger.info(f"Directory created: {non_existent_dir}")
        # Clean up
        os.rmdir(non_existent_dir)
    except Exception as e:
        logger.error(f"Error creating directory: {e}")
    
    # 3. Config validation
    logger.info("Testing config validation...")
    
    valid_config = {
        "model": "test_model",
        "batch_size": 32,
        "learning_rate": 0.001
    }
    
    invalid_config = {
        "model": "test_model",
        "batch_size": "not a number"
    }
    
    required_keys = ["model", "batch_size"]
    
    key_validators = {
        "batch_size": lambda x: isinstance(x, int),
        "learning_rate": lambda x: isinstance(x, float) and x > 0
    }
    
    try:
        validate_config(valid_config, required_keys, key_validators=key_validators)
        logger.info("Validation passed for valid config")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    try:
        validate_config(invalid_config, required_keys, key_validators=key_validators)
    except Exception as e:
        logger.info(f"Expected error caught: {e}")


def test_system_info(logger):
    """Test system information logging."""
    logger.info("=== Testing System Information Logging ===")
    
    # Log system info
    system_info = log_system_info()
    
    # Print some key information
    logger.info(f"Python version: {system_info.get('python_version')}")
    logger.info(f"OS: {system_info.get('os')}")
    logger.info(f"CPU: {system_info.get('cpu')}")
    logger.info(f"Memory: {system_info.get('memory_total_gb')} GB")
    
    if system_info.get('cuda_available'):
        logger.info(f"CUDA version: {system_info.get('cuda_version')}")
        logger.info(f"GPU count: {system_info.get('gpu_count')}")
        for i, gpu in enumerate(system_info.get('gpu_names', [])):
            logger.info(f"GPU {i}: {gpu}")


def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description="Test error handling and logging system")
    parser.add_argument("--test", choices=["all", "basic", "progress", "training", "exceptions", 
                                          "decorators", "validation", "system"], 
                        default="all", help="Which test to run")
    args = parser.parse_args()
    
    logger, log_dir = setup_test_environment()
    logger.info("Starting error handling and logging tests")
    
    tests = {
        "basic": lambda: test_basic_logging(logger),
        "progress": lambda: test_progress_logger(logger, log_dir),
        "training": lambda: test_training_logger(logger, log_dir),
        "exceptions": lambda: test_exception_handling(logger),
        "decorators": lambda: test_decorators(logger),
        "validation": lambda: test_validation_functions(logger),
        "system": lambda: test_system_info(logger)
    }
    
    if args.test == "all":
        for test_name, test_func in tests.items():
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Running {test_name} test")
            logger.info(f"{'=' * 50}\n")
            test_func()
    else:
        tests[args.test]()
    
    logger.info("\nTests completed! Check the logs directory for output files.")


if __name__ == "__main__":
    main()
