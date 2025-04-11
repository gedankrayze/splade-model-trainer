"""
Unified SPLADE Trainer Module

This package provides a comprehensive toolkit for training SPLADE (SParse Lexical AnD Expansion) models
with advanced features including mixed precision training, early stopping, and checkpointing.

Components:
----------
- UnifiedSpladeTrainer: The main trainer class that integrates all advanced features
- SpladeDataset: Enhanced dataset implementation for loading and preprocessing training data
- utils: Utility functions and classes for early stopping, checkpointing, and logging

The unified trainer brings together the best components from previous implementations
into a cohesive, maintainable solution.

Example usage:
-------------
```python
from src.unified import UnifiedSpladeTrainer

# Create trainer
trainer = UnifiedSpladeTrainer(
    model_name="prithivida/Splade_PP_en_v1",
    output_dir="./fine_tuned_splade",
    train_file="data/training.json",
    val_file="data/validation.json",
    learning_rate=5e-5,
    batch_size=8,
    epochs=3,
    use_mixed_precision=True,
    save_best_only=True
)

# Train model
trainer.train()
```

For command-line usage, see the `train_splade_unified.py` script.
"""

from src.unified.trainer import UnifiedSpladeTrainer
from src.unified.dataset import SpladeDataset

__all__ = ['UnifiedSpladeTrainer', 'SpladeDataset']
