# Modular Training Data Generator

This directory contains a modularized version of the training data generator for SPLADE model fine-tuning. The code has been restructured to improve maintainability and add language support.

## Overview

The modular generator is structured as follows:

```
src/
├── generate_training_data_modular.py  # Main entry point
├── generator/
    ├── __init__.py
    ├── api.py          # API client for OpenAI/Ollama
    ├── models.py       # Pydantic data models
    ├── processors.py   # Document loading and processing
    ├── templates.py    # Domain and language templates
    └── utils.py        # Utility functions
```

## Key Features

- **Modular Architecture**: Code is organized into logical modules for better maintainability
- **Multilingual Support**: Generate training data in different languages
- **Language Detection**: Automatically detect document language
- **Language-Specific Templates**: Templates optimized for different languages
- **Improved Error Handling**: Better error messages and recovery

## Language Support

The modular generator supports multilingual training data generation:

- Specify a language with `--language CODE` (e.g., `--language de` for German)
- Automatically detect language with `--detect-language`
- Use language-specific templates for better results
- Currently supports English (`en`) and German (`de`) templates
- Extensible to more languages by adding templates

## Usage Examples

### Generate German Training Data

```bash
task prepare-with-ollama-lang folder=your/german/docs model=llama3 lang=de
```

### Auto-detect Language

```bash
task prepare-with-ollama-auto folder=your/docs model=llama3
```

## Adding Support for More Languages

To add support for a new language:

1. Add templates for the language in `src/generator/templates.py`
2. Add language detection markers in the `detect_document_language` function
3. Add the language name to the `LANGUAGE_NAMES` dictionary

## Command Line Options

The modular generator supports all the original options plus:

- `--language CODE`: Specify language for generated examples (e.g., "en", "de")
- `--detect-language`: Automatically detect document language

## Extending the Generator

To add new functionality:

- Add new domain templates in `templates.py`
- Extend document processing in `processors.py`
- Add new API client options in `api.py`
- Add new utility functions in `utils.py`

## Compatibility

The modular generator is fully compatible with the original one and uses the same output format.
