#!/usr/bin/env python3
"""
Dependency Test for CI/CD pipeline.

This script runs a simple test to verify that core dependencies are working correctly,
which is critical for the CI/CD pipeline to ensure that dependency updates don't break
the codebase.
"""

import os
import sys
import logging
import argparse

# Add the project root to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_dependency_tests():
    """
    Run tests to verify core dependencies.
    """
    success = True
    
    # Test 1: Import NumPy
    try:
        logger.info("Testing NumPy import...")
        import numpy as np
        logger.info(f"NumPy version: {np.__version__}")
    except Exception as e:
        logger.error(f"NumPy import failed: {e}")
        success = False
    
    # Test 2: Import PyTorch
    try:
        logger.info("Testing PyTorch import...")
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Test CUDA/MPS availability
        if torch.cuda.is_available():
            logger.info("CUDA is available")
            logger.info(f"CUDA version: {torch.version.cuda}")
        elif torch.backends.mps.is_available():
            logger.info("MPS (Apple Metal) is available")
        else:
            logger.info("Running on CPU")
    except Exception as e:
        logger.error(f"PyTorch import failed: {e}")
        success = False
    
    # Test 3: Import Transformers
    try:
        logger.info("Testing Transformers import...")
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # Try to load a tokenizer (minimal test)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        logger.info("Successfully loaded tokenizer")
    except Exception as e:
        logger.error(f"Transformers import or tokenizer loading failed: {e}")
        success = False
    
    # Test 4: Import project modules
    try:
        logger.info("Testing project imports...")
        from src.utils import setup_logging
        logger.info("Imported setup_logging")
        
        from src.unified.trainer import UnifiedSpladeTrainer
        logger.info("Imported UnifiedSpladeTrainer")
    except Exception as e:
        logger.error(f"Project import failed: {e}")
        success = False
    
    # Test 5: Simple PyTorch operations
    try:
        logger.info("Testing basic PyTorch operations...")
        x = torch.randn(3, 4)
        y = torch.randn(4, 5)
        z = torch.matmul(x, y)
        logger.info(f"Matrix multiplication result shape: {tuple(z.shape)}")
    except Exception as e:
        logger.error(f"PyTorch operations failed: {e}")
        success = False
    
    return success

def main():
    """
    Main function to parse arguments and run tests.
    """
    parser = argparse.ArgumentParser(description="Test core dependencies for CI/CD")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    success = run_dependency_tests()
    
    if success:
        logger.info("All dependency tests passed!")
        print("\n✅ Dependency tests passed successfully!")
        sys.exit(0)
    else:
        logger.error("Some dependency tests failed!")
        print("\n❌ Dependency tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
