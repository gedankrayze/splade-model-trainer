# SPLADE Model Trainer

A comprehensive toolkit for training, evaluating, and deploying SPLADE (SParse Lexical AnD Expansion) models for
efficient information retrieval.

## Overview

SPLADE is a state-of-the-art approach to information retrieval that combines the efficiency of sparse retrievers with
the effectiveness of neural language models. The SPLADE model uses a sparse representation that captures lexical
matching while also handling term expansion, making it powerful for search applications.

## Project Structure

- `src/` - Source code for the SPLADE model trainer
- `tests/code/` - Unit tests and integration tests
- `docs/` - Documentation for the project
- `articles/` - Articles and blog posts about SPLADE and usage of this toolkit
- `fine_tuned_*/` - Output directories for trained models (not included in version control)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training a Model

```bash
python -m src.train_splade --train-file training_data.json --output-dir ./fine_tuned_splade
```

### Interactive Search

```bash
python -m tests.code.test_queries --model-dir ./fine_tuned_splade --docs-file documents.json
```

## Documentation

For detailed documentation, see the [docs/README.md](docs/README.md) file.

For best practices on training and using SPLADE models, see [docs/best_practices.md](docs/best_practices.md).

## License

See the [LICENSE](LICENSE) file for more details.
