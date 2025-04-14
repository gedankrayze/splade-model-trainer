# Custom Templates for SPLADE Training Data Generation

This document explains how to use and create custom templates for generating training data with the SPLADE Model Trainer.

## Overview

The template system allows you to customize how training data is generated, enabling you to tailor the generation process to specific domains, languages, or use cases. Templates define the system prompts and instructions sent to the large language model when generating training data.

## Using Custom Templates

The new `--template` argument in the data generation tools accepts:

1. Built-in template names (e.g., "generic", "technical", "legal")
2. Paths to custom template files (JSON format)

### Command Line Usage

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

### Using with Task Runner

```bash
# Using a built-in template
task generate input_dir=./documents output_file=training_data.json template=legal

# Using a custom template file
task generate input_dir=./documents output_file=training_data.json template=./templates/my_custom_template.json
```

## Creating Custom Templates

Custom templates are JSON files with specific fields that control the generation process.

### Template Structure

```json
{
  "name": "My Custom Template",
  "language": "en",
  "description": "Template for my specific domain",
  "system_prompt": "You are an expert at creating training data for information retrieval models focused on [DOMAIN].\nYour task is to create realistic search queries that a user might use to find specific information in [DOMAIN] documentation.\n\nFor each query:\n1. Create a natural, specific question a user might search for\n2. Identify the exact text passage that answers this query\n3. Find negative examples - text that looks similar but doesn't answer the query\n\nFocus on [SPECIFIC INSTRUCTIONS FOR YOUR DOMAIN]."
}
```

### Required Fields

- `system_prompt`: The system prompt sent to the LLM for generation

### Optional Fields

- `name`: A human-readable name for the template
- `language`: The language code (e.g., "en", "de") for the template
- `description`: A description of the template's purpose

### Template Best Practices

1. **Be Specific**: Provide detailed instructions for your domain
2. **Include Examples**: Adding examples in your system prompt can improve quality
3. **Specify Terminology**: For domain-specific data, including key terminology helps
4. **Format Instructions**: Include clear formatting instructions if needed

## Example Templates

### E-Commerce Template

```json
{
  "name": "E-Commerce Domain",
  "language": "en",
  "description": "Template for e-commerce product search and discovery",
  "system_prompt": "You are an expert at creating training data for e-commerce search systems.\nYour task is to create realistic search queries that shoppers might use to find specific products or information.\n\nFor each query:\n1. Create a natural, specific search query a shopper might use\n2. Identify the exact product or information passage that answers this query\n3. Find negative examples - products that seem similar but don't actually match the query\n\nFocus on:\n- Product features and specifications\n- Comparative queries (e.g., 'cheaper than', 'better than')\n- Product categories and attributes\n- Customer intent (informational vs. transactional)\n- Short and long-tail queries"
}
```

### Financial Template

```json
{
  "name": "Financial Services Domain",
  "language": "en",
  "description": "Template for financial services and banking information",
  "system_prompt": "You are an expert at creating training data for financial information retrieval systems.\nYour task is to create realistic search queries that customers might use to find specific financial information or services.\n\nFor each query:\n1. Create a natural, specific question a banking customer might search for\n2. Identify the exact text passage that answers this query\n3. Find negative examples - text that looks similar but doesn't answer the query\n\nFocus on:\n- Banking products and services\n- Financial terms and definitions\n- Customer service queries\n- Account management questions\n- Regulatory and compliance information"
}
```

## Language-Specific Templates

You can create language-specific templates by setting the `language` field and adapting the system prompt to the target language:

```json
{
  "name": "German Medical Domain",
  "language": "de",
  "description": "Template for medical information in German",
  "system_prompt": "Du bist ein Experte für die Erstellung von Trainingsdaten für medizinische Informationssysteme.\nDeine Aufgabe ist es, realistische Suchanfragen zu erstellen, die Patienten oder medizinisches Fachpersonal verwenden könnten, um spezifische medizinische Informationen zu finden.\n\nFür jede Anfrage:\n1. Erstelle eine natürliche, spezifische Frage, die jemand suchen könnte\n2. Identifiziere die genaue Textpassage, die diese Anfrage beantwortet\n3. Finde negative Beispiele - Text, der ähnlich aussieht, aber die Anfrage nicht beantwortet\n\nAchte darauf, dass alle Beispiele, Anfragen und Erklärungen auf DEUTSCH sind.\nKonzentriere dich auf medizinische Genauigkeit und fachspezifische Terminologie."
}
```

## Using Templates with the Domain Distiller

Custom templates can also be used with the Domain Distiller tool:

```bash
python -m src.domain_distiller.cli pipeline \
  --template ./templates/my_custom_template.json \
  --language en \
  --queries 100
```

## Template Validation

The system automatically validates custom templates when they're loaded. If required fields are missing, it will fall back to the generic template.

## Advanced Usage: Template Inheritance

You can create more complex templates by implementing a template inheritance system in your custom JSON files:

```json
{
  "name": "Healthcare Regulations",
  "language": "en",
  "description": "Template for healthcare regulatory compliance",
  "base_template": "medical",
  "system_prompt": "You are an expert at creating training data for healthcare regulatory compliance systems.\n[...additional instructions...]"
}
```

## Troubleshooting

If your custom template isn't working as expected:

1. Verify the JSON format is valid
2. Ensure the required `system_prompt` field is included
3. Check that the file path is correct and accessible
4. Review the logs for any error messages about template loading
