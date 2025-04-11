# Domain Distiller: Generate Domain-Specific Training Data Using LLMs

Domain Distiller is a modular, scalable tool for generating high-quality domain-specific training data for SPLADE models using LLMs (Large Language Models). It leverages the knowledge embedded in LLMs to bootstrap domain knowledge, generate realistic queries, and create answer documents - all without requiring existing training data.

## Features

* **Domain Bootstrapping**: Generate comprehensive domain knowledge bases from scratch
* **Multi-Language Support**: Create training data in English, German, Spanish, French, and other languages
* **Contrastive Pair Generation**: Create high-quality "hard negative" examples that improve model training
* **Modular Pipeline**: Run individual steps or the full workflow
* **Asynchronous Processing**: Efficient processing with multiple workers
* **Compatible With Any OpenAI-API Format**: Works with OpenAI, Anthropic, or any compatible API
* **Quality Validation**: Validate and refine generated data
* **Multiple Output Formats**: Export to SPLADE, JSON, JSONL, CSV, or TSV formats

## Installation

The Domain Distiller is included in the SPLADE Model Trainer package. No additional installation is needed.

## Getting Started

### CLI Usage

Domain Distiller provides a command-line interface with various subcommands:

#### Full Pipeline

Generate a complete dataset in one command:

```bash
python -m src.domain_distiller.cli pipeline \
  --domain legal \
  --language en \
  --queries 100 \
  --output-dir ./distilled_data \
  --api-key YOUR_API_KEY \
  --model gpt-4o
```

#### Full Pipeline with Contrastive Generation

Generate a dataset using contrastive pair generation for higher-quality negatives:

```bash
python -m src.domain_distiller.cli pipeline \
  --domain legal \
  --language en \
  --queries 100 \
  --contrastive \
  --output-dir ./distilled_data \
  --api-key YOUR_API_KEY \
  --model gpt-4o
```

#### Individual Steps

You can also run each step separately:

1. Bootstrap domain knowledge:

```bash
python -m src.domain_distiller.cli bootstrap \
  --domain medical \
  --language de \
  --concepts 100 \
  --output-dir ./distilled_data
```

2. Generate queries:

```bash
python -m src.domain_distiller.cli generate-queries \
  --domain-file ./distilled_data/medical_de_domain.json \
  --count 100 \
  --complexity mixed
```

3. Generate documents (standard approach):

```bash
python -m src.domain_distiller.cli generate-documents \
  --queries-file ./distilled_data/medical_de_queries.json \
  --positives-per-query 1 \
  --negatives-per-query 3 \
  --length medium
```

4. Generate documents (with contrastive pairs):

```bash
python -m src.domain_distiller.cli generate-documents \
  --queries-file ./distilled_data/medical_de_queries.json \
  --positives-per-query 1 \
  --negatives-per-query 3 \
  --length medium \
  --contrastive
```

5. Validate dataset:

```bash
python -m src.domain_distiller.cli validate \
  --dataset-file ./distilled_data/medical_de_dataset.json \
  --strict
```

6. Format dataset:

```bash
python -m src.domain_distiller.cli format \
  --dataset-file ./distilled_data/medical_de_dataset.json \
  --format splade \
  --split
```

### Using Custom APIs

Domain Distiller supports any OpenAI-compatible API:

```bash
python -m src.domain_distiller.cli pipeline \
  --domain technical \
  --language en \
  --api-base https://api.anthropic.com \
  --api-key YOUR_KEY \
  --model claude-3-opus-20240229
```

## Domain Templates

Domain Distiller comes with pre-configured templates for common domains:

* Legal
* Medical
* Technical Documentation
* Finance

You can specify a template when bootstrapping a domain:

```bash
python -m src.domain_distiller.cli bootstrap \
  --domain legal \
  --template legal
```

## Language Support

Domain Distiller supports multiple languages:

* English (en)
* German (de)
* Spanish (es)
* French (fr)

Language-specific adaptations ensure natural query patterns and formatting.

## Contrastive Pair Generation

Domain Distiller supports contrastive pair generation, which creates negative documents that are deliberately similar to positive documents but with crucial differences that make them non-relevant to the query. This approach generates higher-quality "hard negatives" that help models learn subtle distinctions.

Contrastive generation employs multiple strategies:

1. **Topical Shift**: Addresses the same general topic but shifts focus to an aspect that doesn't answer the query
2. **Entity Substitution**: Replaces key entities while maintaining similar structure
3. **Temporal Variance**: Changes time frames or sequences that make the document non-responsive
4. **Scope Mismatch**: Provides information that's too general or too specific to answer the query
5. **Premise Alteration**: Changes a fundamental assumption related to the query
6. **Perspective Shift**: Presents information from a different perspective that doesn't address the query

Enable contrastive generation with the `--contrastive` flag:

```bash
python -m src.domain_distiller.cli generate-documents \
  --queries-file legal_en_queries.json \
  --contrastive
```

For more information, see the [contrastive generation documentation](contrastive_generation.md).

## Progressive Refinement

For best results, use the progressive refinement approach:

1. Start with a small test run (10-20 queries)
2. Validate and review the quality
3. Adjust domain templates or parameters
4. Generate a larger dataset

## Advanced Configuration

### Worker Parallelism

Control the number of parallel workers:

```bash
python -m src.domain_distiller.cli pipeline \
  --domain finance \
  --workers 8
```

### Output Formats

Choose from various output formats:

```bash
python -m src.domain_distiller.cli format \
  --dataset-file ./distilled_data/finance_en_dataset.json \
  --format jsonl
```

### Dataset Splitting

Split into train/validation/test sets:

```bash
python -m src.domain_distiller.cli format \
  --dataset-file ./distilled_data/finance_en_dataset.json \
  --format splade \
  --split \
  --train-ratio 0.8 \
  --val-ratio 0.1
```

## Training SPLADE Models with Generated Data

After generating data, you can train a SPLADE model:

```bash
python train_splade_unified.py \
  --train-file ./distilled_data/legal_en_splade_train.json \
  --val-file ./distilled_data/legal_en_splade_val.json \
  --output-dir ./fine_tuned_splade \
  --mixed-precision
```

## Extending Domain Distiller

### Creating Custom Domain Templates

Create a new file in `src/domain_distiller/templates/`:

```python
# src/domain_distiller/templates/ecommerce.py

TEMPLATE = {
    "name": "E-commerce Domain",
    "description": "E-commerce products, categories, and customer reviews",
    "bootstrap_instructions": """Focus on key e-commerce concepts including:
    - Product categories and attributes
    - Customer reviews and sentiment
    - Pricing and promotions
    - ...
    """,
    "query_complexity": {
        "simple": 0.4,
        "intermediate": 0.5,
        "complex": 0.1
    },
    "document_length": {
        "short": 0.3,
        "medium": 0.6,
        "long": 0.1
    }
}
```

### Adding Language Support

Create a new file in `src/domain_distiller/languages/`:

```python
# src/domain_distiller/languages/it.py

LANGUAGE = {
    "name": "Italian",
    "code": "it",
    "description": "Italian language (Italiano)",
    "bootstrap_instructions": "Generate all content in fluent, professional Italian.",
    "query_patterns": [
        "Cosa è {concept}?",
        "Come funziona {process}?",
        # More Italian query patterns...
    ]
}
```

## Troubleshooting

### API Rate Limiting

If you encounter rate limiting errors, try:

1. Reducing the number of workers
2. Using the `--batch-size` parameter to control batch size
3. Adding delays between API calls

### Quality Issues

For quality issues:

1. Try a different LLM model (`--model` parameter)
2. Adjust temperature settings
3. Use the validation step to identify problematic examples
4. Refine domain templates with more specific instructions
5. Enable contrastive generation for higher-quality negatives

## Best Practices

* Start with high-quality domain bootstrapping (more concepts = better data)
* Use complex queries for more realistic training data
* Enable contrastive generation for challenging negative examples
* Validate examples and review quality metrics
* Split data into train/validation/test sets for proper evaluation
* Experiment with different prompt templates for your specific domain

## Technical Details

Domain Distiller uses:

* Asynchronous processing with `asyncio`
* Robust error handling and retry logic
* JSON schema validation for LLM outputs
* Streaming and batching for efficient processing
* Progressive generation for large datasets
* Contrastive strategies for high-quality training data
