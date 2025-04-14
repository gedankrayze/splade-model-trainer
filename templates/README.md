# Custom Templates for SPLADE Training Data Generation

This directory contains custom templates for generating domain-specific training data with the SPLADE Model Trainer.

## Available Templates

| Template File | Domain | Description |
|---------------|--------|-------------|
| cold_chain_management.json | Cold Chain Management | Temperature-controlled supply chains and refrigerated transport |
| cybersecurity_incident_response.json | Cybersecurity | Security incident detection, response, and threat mitigation |
| english_documentation_hvac.json | HVAC (English) | Product documentation for sustainable heating and ventilation systems |
| english_technical_hvac.json | HVAC (English) | Technical specifications for sustainable HVAC systems |
| financial_risk_management.json | Finance | Financial risk assessment and mitigation strategies |
| german_documentation_hvac.json | HVAC (German) | Produktdokumentation für nachhaltige Heiz- und Lüftungssysteme |
| german_technical_hvac.json | HVAC (German) | Technische Spezifikationen für nachhaltige HLK-Systeme |
| healthcare_it.json | Healthcare | Healthcare information systems and medical IT infrastructure |
| renewable_energy_projects.json | Energy | Renewable energy project development and implementation |
| sustainable_manufacturing.json | Manufacturing | Sustainable manufacturing practices and green production |

## Using These Templates

You can use these templates with the data generation tools by providing the template file path:

```bash
# Using the command line
python -m src.generate_training_data \
  --input-dir ./documents \
  --output-file training_data.json \
  --template ./templates/cold_chain_management.json

# Using Task runner
task generate input_dir=./documents output_file=training_data.json \
  template=./templates/cold_chain_management.json
```

You can also use them with the Domain Distiller:

```bash
python -m src.domain_distiller.cli pipeline \
  --template ./templates/healthcare_it.json \
  --language en \
  --queries 100
```

## Template Structure

Each template is a JSON file with the following structure:

```json
{
  "name": "Template Name",
  "language": "en",
  "description": "Brief description of the template domain",
  "system_prompt": "Detailed instructions for the LLM on how to generate data..."
}
```

The most important field is `system_prompt`, which provides instructions to the large language model for generating training data.

## Creating Your Own Templates

To create a new template:

1. Create a new JSON file in this directory with a descriptive name
2. Follow the structure above, focusing on creating a detailed `system_prompt`
3. Include domain-specific terminology, concepts, and examples in your prompt
4. Save the file and use it with the `--template` parameter

For more detailed information, see the [custom templates documentation](../docs/custom_templates.md).
