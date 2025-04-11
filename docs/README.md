# Gedank Rayze SPLADE Model Trainer

A comprehensive toolkit for training, evaluating, and deploying SPLADE (SParse Lexical AnD Expansion) models for
efficient information retrieval. This project provides a complete pipeline for building high-performance sparse
retrieval systems.

## Overview

SPLADE is a state-of-the-art approach to information retrieval that combines the efficiency of sparse retrievers with
the effectiveness of neural language models. The SPLADE model uses a sparse representation that captures lexical
matching while also handling term expansion, making it powerful for search applications.

This toolkit provides:

- Training infrastructure for fine-tuning SPLADE models on domain-specific data
- Evaluation tools for measuring retrieval performance
- Inference utilities for encoding queries and documents
- Support for hybrid search combining dense and sparse embeddings

## Key Features

- **Fine-tuning**: Train SPLADE models on custom datasets
- **Evaluation**: Measure performance using standard IR metrics (MRR, P@k, NDCG)
- **Inference**: Encode queries and documents for retrieval
- **Data generation**: Create training data from document collections
- **Hardware optimization**: Support for CUDA, MPS (Apple Silicon), and CPU

## Use Cases

### 1. Domain-Specific Search

Fine-tune a SPLADE model on your technical documentation to create a specialized search engine:

```bash
# Generate training data from documentation
python generate_training_data.py --input-dir ./docs --output-file training_data.json

# Train the model
python train_splade.py --train-file training_data.json --output-dir ./fine_tuned_splade
```

### 2. Interactive Search

Create an interactive search interface for your document collection:

```bash
# Run the interactive query interface
python test_queries.py --model-dir ./fine_tuned_splade --docs-file documents.json
```

### 3. Evaluation and Benchmarking

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

The project requires Python 3.7+ and the following dependencies:

```
torch
transformers
numpy
tqdm
pydantic
openai (optional, for data generation)
```

To install dependencies:

```bash
pip install torch transformers numpy tqdm pydantic openai
```

## Hardware Support

The toolkit automatically detects and utilizes available hardware acceleration:

- CUDA for NVIDIA GPUs
- MPS for Apple Silicon
- Falls back to CPU when neither is available

# SPLADE Model Trainer: Best Practices

This document outlines optimized approaches and best practices to maximize the efficiency and effectiveness of your
SPLADE model training and deployment workflow.

## Data Preparation

### Document Collection

- **Optimal chunking**: Keep chunks between 500-2000 characters. Too small = insufficient context; too large = diluted
  relevance signals
- **Content overlap**: Include 10-20% overlap between chunks to avoid splitting key information
- **Pre-processing**: Apply consistent normalization (lowercase, punctuation removal, etc.) to both training data and
  inference queries
- **Balance**: Ensure your training data has diverse document types representing your actual retrieval corpus

### Training Data Generation

- **Query diversity**: Generate a variety of query types (questions, keywords, natural language) that reflect real user
  behavior
- **Hard negatives**: Include semantically similar but incorrect documents as negative examples
- **Validation split**: Maintain a 80/20 or 90/10 train/validation split with minimal content overlap
- **Quality over quantity**: 1,000 high-quality examples often outperform 10,000 poor ones

## Model Training

### Hyperparameter Optimization

- **Regularization coefficients**: Start with low values (`lambda_d` = `lambda_q` = 0.0001) and adjust based on
  validation loss
- **Learning rate**: 2e-5 to 5e-5 typically works well for fine-tuning
- **Batch size**: Use the largest that fits in memory (typically 8-32) for more stable gradients
- **Epochs**: 3-5 epochs are usually sufficient; use early stopping based on validation loss

### Computational Efficiency

- **Mixed precision training**: Enable FP16 training to reduce memory usage and increase throughput
- **Gradient accumulation**: If hardware-limited, accumulate gradients over multiple batches
- **Progressive batch size**: Start with a smaller batch size and increase as training progresses
- **Checkpoint selectively**: Save checkpoints based on validation metrics rather than at fixed intervals

## Evaluation and Tuning

### Metrics to Prioritize

- **Recall@k**: Measures ability to retrieve relevant documents in top-k results
- **NDCG@10**: Evaluates ranking quality for top results
- **MRR**: Mean Reciprocal Rank indicates average position of first relevant result
- **Query latency**: Critical for production systems; target under 100ms per query

### Iterative Improvement

- **Error analysis**: Manually review cases where the model performs poorly
- **Data augmentation**: Add examples targeting model weaknesses
- **Query reformulation**: Add training data with different phrasings for the same information need

## Deployment Considerations

### Efficiency Optimizations

- **Quantization**: Quantize model weights to INT8 for faster inference with minimal quality loss
- **Vocabulary pruning**: Remove unused vocabulary tokens to reduce model size
- **Batch processing**: Process queries in batches when possible
- **Document pre-encoding**: Pre-compute and store document vectors for lookup at query time

### Scaling

- **Indexing strategy**: Use inverted indices for efficient retrieval with sparse representations
- **Hybrid retrieval**: Combine SPLADE (recall) with a cross-encoder reranker (precision) for best results
- **Distributed inference**: Shard document index across multiple servers for handling large collections

## Integration Tips

### Pre and Post Processing

- **Query expansion**: Apply simple rule-based expansion for common abbreviations or synonyms
- **Document segmentation**: Consider semantic segmentation rather than fixed-length chunks
- **Filtering**: Apply metadata filters before semantic search to reduce candidate pool

### Monitoring and Maintenance

- **A/B testing**: Compare model versions on a subset of traffic before full deployment
- **Relevance feedback**: Collect user feedback to identify areas for improvement
- **Periodic retraining**: Fine-tune on new data every 3-6 months to capture distribution shifts

## Hardware-Specific Optimizations

### GPU Acceleration

- **CUDA optimizations**: Enable CUDA graph optimization for repeated operations
- **Batch size tuning**: Find optimal batch size for your specific GPU model
- **Memory management**: Monitor GPU memory usage and adjust max_length accordingly

### Apple Silicon (MPS)

- **M-series optimization**: When using MPS backend, smaller batch sizes (4-8) often perform better
- **Mixed precision**: Use float16 precision for significant speedups on Apple Silicon

### CPU Deployment

- **Thread optimization**: Set appropriate OMP_NUM_THREADS and MKL_NUM_THREADS environment variables
- **Quantization**: Use INT8 quantization for up to 4x speedup with minimal quality loss
- **Batch size**: Smaller batches (2-4) typically work better on CPU

By following these best practices, you can significantly improve both the efficiency and effectiveness of your SPLADE
model training and deployment pipeline.