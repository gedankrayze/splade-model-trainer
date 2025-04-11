# Gedank Rayze SPLADE Model Trainer

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

## New Features

### Domain Distiller

The new Domain Distiller tool allows you to generate domain-specific training data for SPLADE models from scratch using LLMs. Key features include:

- **Zero-Shot Training Data Generation**: Create training data for any domain without pre-existing datasets
- **Domain Bootstrapping**: Automatically generate domain knowledge, terminology, and concepts
- **Contrastive Pair Generation**: Create high-quality negative examples using advanced contrastive strategies
- **Multi-Language Support**: Generate data in English, German, Spanish, French, and more
- **OpenAI-Compatible API Support**: Works with OpenAI, Anthropic, or any compatible API endpoint

Quick start with Domain Distiller:

```bash
# Generate domain-specific training data
python -m src.domain_distiller.cli pipeline --domain legal --language en --queries 100 --contrastive

# Train a SPLADE model with the generated data
python train_splade_unified.py --train-file ./distilled_data/legal_en_splade.json --output-dir ./fine_tuned_splade
```

See [docs/domain_distiller.md](docs/domain_distiller.md) for detailed documentation and [docs/contrastive_generation.md](docs/contrastive_generation.md) for information about contrastive pair generation.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training a Model with the Unified Trainer (Recommended)

```bash
python train_splade_unified.py --train-file training_data.json --output-dir ./fine_tuned_splade --mixed-precision
```

This unified trainer combines all advanced features in a single, cohesive interface. See [docs/unified_trainer.md](docs/unified_trainer.md) for details.

### Alternative Training Methods

```bash
python -m src.train_splade_mixed_precision --train-file training_data.json --output-dir ./fine_tuned_splade --mixed-precision
```

### Interactive Search

```bash
python -m tests.code.test_queries --model-dir ./fine_tuned_splade --docs-file documents.json
```

## Documentation

For detailed documentation, see the [docs/README.md](docs/README.md) file.

For best practices on training and using SPLADE models, see [docs/best_practices.md](docs/best_practices.md).

For the unified trainer documentation, see [docs/unified_trainer.md](docs/unified_trainer.md).

For information about our CI/CD setup and GitHub Actions workflows, see [docs/ci-cd/github-actions.md](docs/ci-cd/github-actions.md).

For details on the Domain Distiller tool, see [docs/domain_distiller.md](docs/domain_distiller.md).

## License

See the [LICENSE](LICENSE) file for more details.

## Contact

- GitHub: [https://github.com/gedankrayze/splade-model-trainer](https://github.com/gedankrayze/splade-model-trainer)
- Email: info@gedankrayze.com
