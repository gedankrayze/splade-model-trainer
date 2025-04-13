# Multilingual SPLADE Training Data Generation

This document describes how to use the multilingual features of the SPLADE training data generator.

## Overview

The SPLADE model trainer supports generating training data in multiple languages. This is achieved by using a modular training data generator that supports:

1. Specifying the language for generation
2. Automatic language detection
3. Language-specific templates for better results

## Usage

### Generate Data in a Specific Language

To generate training data in a specific language (e.g., German):

```bash
task prepare-with-ollama-lang folder=your/german/docs model=qwen2.5 lang=de
```

Available language codes:
- `en`: English
- `de`: German
- Add other language codes as templates are added

### Automatic Language Detection

To automatically detect the language of the documents and generate training data in that language:

```bash
task prepare-with-ollama-auto folder=your/docs model=qwen2.5
```

The generator will:
1. Sample some text from the documents
2. Detect the language based on common words
3. Use the appropriate language template
4. Generate training data in the detected language

## How It Works

The language support works in two ways:

1. **Language-Specific Templates**: Pre-translated templates in each language that guide the LLM to generate in that language
2. **Explicit Language Instructions**: Clear instructions to generate in the specified language

For German documents, the generator uses German system prompts that instruct the model to generate all examples in German. This produces more consistent results than just telling the model to "generate in German" while using English templates.

## Benefits

Using language-specific generation provides several benefits:

1. **Consistent Language**: All queries, documents, and explanations in the same language
2. **Better Quality**: Native language templates produce more natural language examples
3. **Improved Training**: More realistic search scenarios for the target language

## Technical Details

The language support is implemented in the modular training data generator at:
- `src/generate_training_data_modular.py`
- `src/generator/templates.py` (language templates)

To add support for a new language:
1. Add templates for the language in `templates.py`
2. Add language detection markers 
3. Update the Taskfile if needed

## Troubleshooting

If you encounter mixed language training data:
1. Make sure you're using the modular generator (`generate_training_data_modular.py`)
2. Verify the language parameter is set correctly
3. Try a model with better multilingual capabilities (like qwen2.5)
