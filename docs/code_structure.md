# SPLADE Model Trainer Code Structure

This document provides an overview of the SPLADE Model Trainer code structure and organization.

## Directory Structure

```
splade-model-trainer/
├── src/                     # Source code
│   ├── __init__.py          # Package initialization
│   ├── embedder.py          # SPLADE, Text, and Hybrid embedders
│   ├── train_splade.py      # SPLADE model training
│   ├── train_embeddings.py  # Dense embeddings training
│   ├── train_embeddings_lowmem.py # Low-memory version for dense model training
│   ├── evaluate_splade.py   # SPLADE model evaluation
│   ├── evaluate_embeddings.py # Dense embeddings evaluation
│   └── generate_training_data.py # Training data generation utilities
│
├── tests/                   # Tests
│   ├── code/                # Test code
│   │   ├── __init__.py      # Test package initialization
│   │   ├── test_embedder.py # Tests for embedder functionality
│   │   ├── test_load_model.py # Tests for model loading
│   │   └── test_queries.py  # Interactive query testing
│   │
│   └── __init__.py          # Tests package initialization
│
├── docs/                    # Documentation
│   ├── README.md            # Full documentation
│   ├── best_practices.md    # Best practices for SPLADE
│   └── code_structure.md    # This document
│
├── articles/                # Articles and blog posts
│
├── fine_tuned_*/            # Output directories (not in version control)
│
├── README.md                # Project overview
├── requirements.txt         # Project dependencies
├── LICENSE                  # License information
└── Taskfile.yaml            # Task definitions
```

## Module Overview

### Source Code (`src/`)

- **embedder.py**: Contains classes for creating and managing different types of embeddings:
  - `SpladeEmbedder`: Text embedder using custom SPLADE model for sparse embeddings
  - `TextEmbedder`: Text embedder using FastEmbed dense models
  - `HybridEmbedder`: Combines dense and sparse embeddings for hybrid search
  - `OpenAiTextEmbedder`: Text embedder using OpenAI-compatible embedding APIs
  - `EmbedderFactory`: Factory class to create appropriate embedder based on configuration

- **train_splade.py**: SPLADE model training functionality:
  - `SpladeDataset`: Dataset class for SPLADE training with query-document pairs
  - `SpladeTrainer`: Trainer for SPLADE model fine-tuning

- **train_embeddings.py**: Dense embeddings training functionality:
  - Training infrastructure for fine-tuning dense embedding models

- **train_embeddings_lowmem.py**: Low-memory version of dense training:
  - Optimized for training on hardware with limited memory

- **evaluate_splade.py**: SPLADE model evaluation:
  - Tools for evaluating SPLADE model performance using standard IR metrics

- **evaluate_embeddings.py**: Dense embeddings evaluation:
  - Tools for evaluating dense embedding model performance

- **generate_training_data.py**: Training data generation:
  - Utilities for creating training data from document collections

### Tests (`tests/code/`)

- **test_embedder.py**: Tests for embedder functionality
  - Test cases for all embedding types (SPLADE, dense, hybrid)

- **test_load_model.py**: Tests for model loading
  - Verification that models can be loaded and used properly

- **test_queries.py**: Interactive query testing
  - Command-line interface for testing queries against documents
  - Useful for manual evaluation and demonstration

## Usage Patterns

### Embedding Document Collections

```python
from src.embedder import EmbedderFactory

# Create a SPLADE embedder
embedder = EmbedderFactory.create_embedder(
    embedder_type="splade",
    splade_model_dir="./fine_tuned_splade"
)

# Generate embeddings
documents = ["Document content here", "Another document"]
embeddings = embedder.generate_sparse_embeddings(documents)
```

### Training a New Model

```python
from src.train_splade import SpladeTrainer

# Initialize trainer
trainer = SpladeTrainer(
    model_name="prithivida/Splade_PP_en_v1",
    output_dir="./fine_tuned_splade",
    train_file="training_data.json",
    val_file="validation_data.json",
    learning_rate=5e-5,
    batch_size=8,
    epochs=3
)

# Train model
trainer.train()
```

### Evaluating Models

```python
from src.evaluate_splade import evaluate_model

results = evaluate_model(
    model_dir="./fine_tuned_splade",
    test_file="test_queries.json",
    metrics=["mrr", "ndcg@10", "recall@100"]
)
```

## Extending the Code

The codebase is designed to be extensible:

1. **New Embedder Types**: Extend the base embedder classes or create new ones and register them with the EmbedderFactory.

2. **Custom Training Routines**: Create specialized training routines by extending the SpladeTrainer class.

3. **New Evaluation Metrics**: Add new metrics to the evaluation tools.

4. **Data Processing**: Extend the data generation utilities for specialized document processing.

## Code Conventions

- **Typing**: The codebase uses Python type hints throughout for better IDE integration and error catching.

- **Logging**: Consistent logging patterns are used throughout the code.

- **Exception Handling**: Functions that could fail (like file I/O or model initialization) have proper exception handling.

- **Documentation**: All classes and functions have docstrings in Google Style format.

- **Configuration**: Command-line tools use argparse for configuration.
