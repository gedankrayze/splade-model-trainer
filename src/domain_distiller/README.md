# Domain Distiller

A tool for generating domain-specific training data using LLMs.

## Overview

Domain Distiller is a modular system for generating high-quality training data for SPLADE models without requiring existing datasets. It leverages large language models (LLMs) to create domain-specific knowledge, queries, and documents through a multi-stage process.

## Modules

- **bootstrapper.py**: Generates domain knowledge base
- **query_generator.py**: Creates realistic search queries
- **document_generator.py**: Produces positive and negative documents
- **validator.py**: Validates the quality of generated examples
- **formatter.py**: Formats data for SPLADE training or other formats
- **cli.py**: Command-line interface for all functionality

### Utilities
- **api_utils.py**: Client for OpenAI-compatible APIs
- **logging_utils.py**: Logging configuration

### Domain Templates and Language Support
- **templates/**: Domain-specific templates and settings
- **languages/**: Language-specific configurations

## Usage

Use the CLI for both the full pipeline and individual components:

```bash
# Full pipeline
python -m src.domain_distiller.cli pipeline --domain legal --language en --queries 100

# Full pipeline with contrastive pair generation
python -m src.domain_distiller.cli pipeline --domain legal --language en --queries 100 --contrastive

# Individual steps
python -m src.domain_distiller.cli bootstrap --domain medical --language de
python -m src.domain_distiller.cli generate-queries --domain-file medical_de_domain.json
python -m src.domain_distiller.cli generate-documents --queries-file medical_de_queries.json --contrastive
```

See the full documentation in `/docs/domain_distiller.md` for detailed usage instructions.

## Features

- Create domain-specific training data from scratch
- Multi-language support (English, German, Spanish, French)
- Contrastive pair generation for higher-quality training data
- Asynchronous processing with multiple workers
- Compatible with any OpenAI-API compliant service
- Comprehensive validation and formatting

## Advanced Features

### Contrastive Pair Generation

Domain Distiller supports contrastive pair generation, which creates negative documents that are deliberately similar to positive documents but with crucial differences that make them non-relevant to the query. This approach generates higher-quality "hard negatives" that help models learn subtle distinctions.

To use contrastive pair generation:

```bash
python -m src.domain_distiller.cli generate-documents --queries-file queries.json --contrastive
```

See `/docs/contrastive_generation.md` for more information.

### Multi-Language Support

Generate training data in multiple languages with language-specific configurations:

```bash
python -m src.domain_distiller.cli pipeline --domain legal --language de
```

### Domain Templates

Use pre-configured domain templates:

```bash
python -m src.domain_distiller.cli bootstrap --domain medical --template medical
```

## Requirements

- Python 3.8+
- aiohttp
- tenacity
- The same dependencies as the main SPLADE trainer
