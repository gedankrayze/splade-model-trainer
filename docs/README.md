# Gedank Rayze SPLADE Model Trainer Documentation

A comprehensive toolkit for training, evaluating, and deploying SPLADE (SParse Lexical AnD Expansion) models for
efficient information retrieval. This project provides a complete pipeline for building high-performance sparse
retrieval systems.

## Overview

SPLADE is a state-of-the-art approach to information retrieval that combines the efficiency of sparse retrievers with
the effectiveness of neural language models. The SPLADE model uses a sparse representation that captures lexical
matching while also handling term expansion, making it powerful for search applications.

This toolkit provides:

- Training infrastructure for fine-tuning SPLADE models on domain-specific data
- Domain-specific training data generation using LLMs
- Evaluation tools for measuring retrieval performance
- Inference utilities for encoding queries and documents
- Support for hybrid search combining dense and sparse embeddings

## Documentation Index

- [Unified Trainer](unified_trainer.md) - Documentation for the unified trainer interface
- [Best Practices](best_practices.md) - Best practices for training and using SPLADE models
- [Domain Distiller](domain_distiller.md) - Guide to generating domain-specific training data
- [Contrastive Generation](contrastive_generation.md) - Information about contrastive pair generation
- [CI/CD](ci-cd/github-actions.md) - Information about our CI/CD setup and GitHub Actions workflows

## Key Features

- **Fine-tuning**: Train SPLADE models on custom datasets
- **Domain Distiller**: Generate domain-specific training data from scratch using LLMs
- **Contrastive Learning**: Create high-quality training data using contrastive pair generation
- **Multi-Language Support**: Generate training data in multiple languages
- **Evaluation**: Measure performance using standard IR metrics (MRR, P@k, NDCG)
- **Inference**: Encode queries and documents for retrieval
- **Hardware optimization**: Support for CUDA, MPS (Apple Silicon), and CPU

## Use Cases

### 1. Zero-Shot Domain-Specific Search

Create a domain-specific search engine without pre-existing training data:

```bash
# Generate domain-specific training data using Domain Distiller
python -m src.domain_distiller.cli pipeline --domain legal --language en --queries 100 --contrastive

# Train the model
python train_splade_unified.py --train-file ./distilled_data/legal_en_splade.json --output-dir ./fine_tuned_splade
```

### 2. Custom Training Data Generation

Generate training data from your own documentation:

```bash
# Generate training data from documentation
python -m src.generate_training_data.py --input-dir ./docs --output-file training_data.json

# OR use Domain Distiller with your domain knowledge
python -m src.domain_distiller.cli bootstrap --domain technical --language en --concepts 100
```

### 3. Interactive Search

Create an interactive search interface for your document collection:

```bash
# Run the interactive query interface
python test_queries.py --model-dir ./fine_tuned_splade --docs-file documents.json
```

### 4. Evaluation and Benchmarking

Evaluate model performance on your test dataset:

```bash
# Evaluate retrieval performance
python test_evaluate.py --model-dir ./fine_tuned_splade --test-file validation_data.json
```

## Code Examples

### Training a SPLADE Model

```python
from train_splade import SpladeTrainer

# Initialize trainer
trainer = SpladeTrainer(
    model_name="prithivida/Splade_PP_en_v1",  # Base model
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

### Generating Domain-Specific Training Data

```python
from src.domain_distiller.bootstrapper import bootstrap_domain
from src.domain_distiller.query_generator import generate_queries
from src.domain_distiller.document_generator import generate_documents

# Bootstrap domain knowledge
domain_data = bootstrap_domain(
    domain="medical",
    language="en",
    num_concepts=50
)

# Generate queries
queries = generate_queries(
    domain_file="medical_en_domain.json",
    count=100
)

# Generate documents with contrastive pairs
dataset = generate_documents(
    queries_file="medical_en_queries.json",
    contrastive=True
)
```

### Encoding Documents with SPLADE

```python
from embedder import SpladeEmbedder

# Initialize embedder
embedder = SpladeEmbedder(model_dir="./fine_tuned_splade")

# Generate sparse embeddings
documents = ["Document content here", "Another document"]
embeddings = embedder.generate_sparse_embeddings(documents)
```

### Searching with a Trained Model

```python
from test_queries import load_splade_model, retrieve_documents

# Load model
tokenizer, model, device = load_splade_model("./fine_tuned_splade")

# Load documents
documents = ["First document content", "Second document content", "Third document content"]

# Perform search
query = "What is the capital of France?"
results = retrieve_documents(query, documents, tokenizer, model, device, top_k=3)

# Display results
for result in results:
    print(f"Rank {result['rank']} (Score: {result['score']:.4f}):")
    print(f"  {result['snippet']}")
```

## Installation and Requirements

The project requires Python 3.8+ and the following dependencies:

```
torch
transformers
numpy
tqdm
pydantic
aiohttp
tenacity
openai (optional, for data generation)
```

To install dependencies:

```bash
pip install -r requirements.txt
```

## Hardware Support

The toolkit automatically detects and utilizes available hardware acceleration:

- CUDA for NVIDIA GPUs
- MPS for Apple Silicon
- Falls back to CPU when neither is available
