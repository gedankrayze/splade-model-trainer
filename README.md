# Gedank Rayze SPLADE Model Trainer

A comprehensive toolkit for training, evaluating, and deploying SPLADE (SParse Lexical AnD Expansion) models for
efficient information retrieval.

## Overview

SPLADE is a state-of-the-art approach to information retrieval that combines the efficiency of sparse retrievers with
the effectiveness of neural language models. The SPLADE model uses a sparse representation that captures lexical
matching while also handling term expansion, making it powerful for search applications.

While our primary focus is on SPLADE models, we also provide complementary support for training dense embedding models and hybrid approaches that can be used alongside SPLADE for certain use cases.

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

### Custom Templates

The toolkit now supports custom templates for generating domain-specific training data:

```bash
# Using a built-in template
python -m src.generate_training_data \
  --input-dir ./documents \
  --output-file training_data.json \
  --template legal

# Using a custom template file
python -m src.generate_training_data \
  --input-dir ./documents \
  --output-file training_data.json \
  --template ./templates/my_custom_template.json
```

See [docs/custom_templates.md](docs/custom_templates.md) for detailed documentation on creating and using custom templates.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training a Model with the Unified Trainer

```bash
python train_splade_unified.py --train-file training_data.json --output-dir ./fine_tuned_splade --mixed-precision
```

The unified trainer provides a comprehensive solution that uses tools from the `src/unified` folder, combining all advanced features in a single, cohesive interface. It offers:

- Mixed precision training for better performance
- Early stopping to prevent overfitting
- Checkpointing for saving/resuming training
- Training recovery options
- Comprehensive logging and metrics tracking
- Support for multiple hardware platforms (CUDA, MPS, CPU)

See [docs/unified_trainer.md](docs/unified_trainer.md) for detailed documentation and advanced options.

### Using Task Runner with Enhanced Documentation

We provide extensively documented Taskfiles that simplify common operations and automatically handle virtual environment activation:

```bash
# Install Task runner: https://taskfile.dev/installation/

# Generate training data with a custom template
task generate input_dir=./documents output_file=training.json template=legal language=de

# Train a model with the generated data
task train train_file=training.json output_dir=./fine_tuned_splade

# Generate language-specific data using OpenAI
task train:prepare-with-openai-lang folder=./documents model=gpt-4o lang=es template=legal
```

Each task comes with detailed documentation and examples. Use `task -l` to list all available tasks.

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
