#!/usr/bin/env python3
"""
Mini training test for CI/CD pipeline.

This script runs a very small training job with minimal data to verify that 
the core training functionality works as expected with current dependencies.
It's designed to be fast and lightweight for CI environments.
"""

import os
import sys
import json
import logging
import tempfile
import argparse
import torch
from typing import Dict, List, Any

# Add the project root to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.utils import setup_logging
from src.unified.utils import TrainingLogger
from src.unified.trainer import UnifiedSpladeTrainer

def create_sample_data() -> List[Dict[str, Any]]:
    """
    Create a minimal dataset for testing training.
    """
    return [
        {
            "query": "how to make pasta",
            "positive_document": "Cook pasta in boiling water until al dente. Drain and serve with sauce.",
            "negative_documents": ["Pasta is a type of noodle made from flour, water, and sometimes eggs."]
        },
        {
            "query": "python programming basics",
            "positive_document": "Python is a high-level programming language known for its readability and simplicity.",
            "negative_documents": ["Snake species are found on every continent except Antarctica."]
        },
        {
            "query": "climate change effects",
            "positive_document": "Rising global temperatures contribute to changing precipitation patterns and extreme weather events.",
            "negative_documents": ["The climate of a region is determined by its latitude, terrain, and altitude."]
        },
        {
            "query": "healthy breakfast ideas",
            "positive_document": "Oatmeal with fruits and nuts is a nutritious way to start your day.",
            "negative_documents": ["Breakfast is the first meal of the day."]
        }
    ]

def run_mini_training() -> bool:
    """
    Run a minimal training job and return True if it completes successfully.
    """
    # Set up logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_dir=log_dir, level="INFO")
    training_logger = TrainingLogger("mini_training_test", log_dir)
    
    logger.info("Starting mini training test")
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as output_dir:
        # Create a temporary file for the training data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            json.dump(create_sample_data(), temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Configure a minimal training run
            trainer = UnifiedSpladeTrainer(
                model_name="distilbert-base-uncased",
                output_dir=output_dir,
                train_file=temp_file_path,
                learning_rate=5e-5,
                batch_size=2,
                epochs=1,
                max_length=128,
                lambda_d=0.0001,
                lambda_q=0.0001,
                use_mixed_precision=False,  # Disable for CI compatibility
                early_stopping_patience=1,
                save_best_only=True,
                save_freq=1,
                max_checkpoints=1,
                logger=training_logger  # Use the training logger here
            )
            
            # Run training
            trainer.train()
            
            # Check if model files were created in the final_model directory
            final_model_dir = os.path.join(output_dir, "final_model")
            
            # List files and check for model files
            logger.info(f"Checking final model directory: {final_model_dir}")
            if os.path.exists(final_model_dir):
                files = os.listdir(final_model_dir)
                logger.info(f"Files in final model directory: {files}")
                
                # Check for model files (any of these would indicate success)
                model_files = [
                    "pytorch_model.bin", 
                    "model.safetensors",
                    "config.json",
                    "tokenizer_config.json"
                ]
                
                has_model_files = any(mf in files for mf in model_files)
                logger.info(f"Found model files: {has_model_files}")
            else:
                logger.error(f"Final model directory does not exist: {final_model_dir}")
                has_model_files = False
            
            # Clean up
            os.unlink(temp_file_path)
            
            return has_model_files
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            return False

def main():
    """
    Main function that parses arguments and runs the test.
    """
    parser = argparse.ArgumentParser(description="Run a mini training test for CI/CD")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    if args.verbose:
        setup_logging(level="DEBUG")
    
    success = run_mini_training()
    
    if success:
        print("✅ Mini training test passed!")
        sys.exit(0)
    else:
        print("❌ Mini training test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
