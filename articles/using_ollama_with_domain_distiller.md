# Create a Custom Cold-Chain Search System Using Ollama and Domain Distiller

In this guide, we'll walk through the process of building a domain-specific search system for cold-chain management and logistics using locally hosted models with Ollama. Instead of relying on expensive API calls to commercial services, we'll use the open-source Qwen 2.5 model to generate our training data, then fine-tune a SPLADE model for powerful, domain-specific search capabilities.

We'll demonstrate how to create separate search capabilities for both English and Dutch by running the process for each language individually—a key feature of Domain Distiller that allows for language-specific optimization.

## Prerequisites

- A computer with at least 16GB RAM and 50GB free disk space
- Basic familiarity with command line interfaces
- Python 3.8+ installed
- Git installed

## Step 1: Setting Up Your Environment

First, let's set up Ollama and install the necessary tools:

### Install Ollama

Follow the instructions at [Ollama.com](https://ollama.com/) to install Ollama for your operating system.

Once installed, pull the Qwen 2.5 model:

```bash
ollama pull qwen2.5
```

### Clone the SPLADE Model Trainer Repository

```bash
git clone https://github.com/gedankrayze/splade-model-trainer.git
cd splade-model-trainer
```

### Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Configure Domain Distiller to Use Ollama

The Domain Distiller tool is designed to work with any OpenAI-compatible API. Ollama provides such an API, allowing us to use locally hosted models instead of commercial services.

First, start the Ollama server:

```bash
ollama serve
```

By default, this runs the server at `http://localhost:11434`.

## Step 3: Creating a Domain Template for Cold-Chain Logistics

Let's create a custom domain template to guide our training data generation:

```bash
mkdir -p src/domain_distiller/templates
```

Create a new file `src/domain_distiller/templates/cold_chain.py`:

```python
"""
Cold-chain logistics domain template for the Domain Distiller.
"""

TEMPLATE = {
    "name": "Cold Chain Logistics",
    "description": "Cold chain management and logistics for temperature-sensitive products",
    "bootstrap_instructions": """Focus on key cold chain logistics concepts including:
- Temperature control systems and monitoring
- Refrigerated transport and storage
- Thermal packaging solutions
- Temperature-sensitive product handling
- Cold chain validation and qualification
- Regulatory compliance for cold chain
- Data logging and real-time monitoring
- Cold chain risk management
- Last-mile delivery for temperature-sensitive items
- Pharmaceutical cold chain requirements
- Food cold chain management
- Vaccine distribution logistics

Include specialized terminology for equipment, processes, regulations, and quality assurance in cold chain logistics.

For document types, include:
- Standard Operating Procedures (SOPs)
- Temperature excursion reports
- Cold chain validation protocols
- Quality agreements
- Transport qualification documents
- Regulatory guidelines
- Equipment specifications
- Risk assessment documents
- Temperature mapping studies

For relationships, focus on the connections between different cold chain components,
temperature requirements for different product types, regulatory frameworks across regions,
and escalation procedures for temperature excursions.
""",
    "query_complexity": {
        "simple": 0.3,  # 30% simple queries
        "intermediate": 0.5,  # 50% intermediate queries
        "complex": 0.2  # 20% complex queries
    },
    "document_length": {
        "short": 0.2,  # 20% short documents
        "medium": 0.6,  # 60% medium documents
        "long": 0.2  # 20% long documents
    },
    "query_templates": [
        "What are the requirements for {cold_chain_concept}?",
        "How to manage {temperature_issue} in cold chain?",
        "What is the difference between {cold_chain_concept_1} and {cold_chain_concept_2}?",
        "Best practices for {cold_chain_process}",
        "How to prevent {cold_chain_failure}?",
        "What temperature range is required for {temperature_sensitive_product}?",
        "Regulations for {cold_chain_activity} in {region}",
        "How to validate {cold_chain_equipment}?",
        "Common issues with {cold_chain_component} and solutions",
        "How to document {cold_chain_process}?"
    ]
}
```

## Step 4: Generate Training Data Using Domain Distiller with Ollama

### Important Note on Multilingual Support

Domain Distiller supports multiple languages, but requires running separate processes for each language. This is by design, as it allows for:

1. Language-specific optimization of the training data
2. Specialized models that understand the linguistic nuances of each language
3. Better performance compared to a single multilingual model

We'll run the process separately for English and Dutch to create two specialized models.

### English Training Data Generation

First, let's generate training data for English:

```bash
python -m src.domain_distiller.cli pipeline \
  --domain cold_chain \
  --language en \
  --queries 100 \
  --contrastive \
  --api-base http://localhost:11434/v1 \
  --model qwen2.5 \
  --output-dir ./distilled_data_coldchain
```

This command:
- Uses our `cold_chain` domain template
- Generates data in English (`--language en`)
- Creates 100 query-document pairs
- Applies contrastive pair generation for higher-quality training
- Uses the local Ollama server with Qwen 2.5
- Saves the output to a dedicated directory

### Dutch Training Data Generation

After the English generation completes, we'll run a separate process for Dutch:

```bash
python -m src.domain_distiller.cli pipeline \
  --domain cold_chain \
  --language nl \
  --queries 100 \
  --contrastive \
  --api-base http://localhost:11434/v1 \
  --model qwen2.5 \
  --output-dir ./distilled_data_coldchain
```

This command is identical to the English one except for the language parameter (`--language nl`).

### Understanding the Generated Data

Let's examine what was created:

```bash
ls -la ./distilled_data_coldchain/
```

You should see separate sets of files for each language:

For English:
- `cold_chain_en_domain.json` - Domain knowledge base in English
- `cold_chain_en_queries.json` - Generated queries in English
- `cold_chain_en_contrastive_dataset.json` - Dataset with positive and negative documents
- `cold_chain_en_splade.json` - Formatted data ready for SPLADE training

For Dutch:
- `cold_chain_nl_domain.json` - Domain knowledge base in Dutch
- `cold_chain_nl_queries.json` - Generated queries in Dutch
- `cold_chain_nl_contrastive_dataset.json` - Dataset with positive and negative documents
- `cold_chain_nl_splade.json` - Formatted data ready for SPLADE training

## Step 5: Training Separate SPLADE Models for Each Language

Now that we have training data for both languages, we'll train separate SPLADE models for each. Using distinct models for each language ensures optimal performance as each model can specialize in the linguistic patterns of its target language.

### English Model Training

```bash
python train_splade_unified.py \
  --base-model prithivida/Splade_PP_en_v2 \
  --train-file ./distilled_data_coldchain/cold_chain_en_splade.json \
  --output-dir ./fine_tuned_splade_coldchain_en \
  --epochs 3 \
  --learning-rate 2e-5 \
  --batch-size 8 \
  --mixed-precision
```

### Dutch Model Training

Once the English model training completes, train the Dutch model:

```bash
python train_splade_unified.py \
  --base-model prithivida/Splade_PP_en_v2 \
  --train-file ./distilled_data_coldchain/cold_chain_nl_splade.json \
  --output-dir ./fine_tuned_splade_coldchain_nl \
  --epochs 3 \
  --learning-rate 2e-5 \
  --batch-size 8 \
  --mixed-precision
```

Note that we're using the same base model (`prithivida/Splade_PP_en_v2`) for both languages, but training it on language-specific data. The fine-tuning process adapts the model to each language's unique characteristics.

Training will take some time depending on your hardware. With a decent CPU, expect 1-3 hours per model.

## Step 6: Testing Your Language-Specific SPLADE Models

Once training is complete, you'll have two separate models—one for English and one for Dutch. You can test each model individually:

### English Model Testing

```bash
python -m tests.code.test_queries \
  --model-dir ./fine_tuned_splade_coldchain_en \
  --interactive
```

Try some English queries like:
- "What temperature range is needed for frozen vaccines?"
- "How to handle temperature excursions during transport?"
- "Best practices for last-mile delivery in cold chain"

### Dutch Model Testing

```bash
python -m tests.code.test_queries \
  --model-dir ./fine_tuned_splade_coldchain_nl \
  --interactive
```

Try some Dutch queries like:
- "Welke temperatuur is vereist voor bevroren medicijnen?"
- "Hoe om te gaan met temperatuurafwijkingen tijdens transport?"
- "Beste praktijken voor last-mile bezorging in de koudeketen"

## Step 7: Integrating With Your Document Collection

You'll need to decide how to handle multilingual search in your application. Here are two common approaches:

### Approach 1: Language-Specific Indices

Create separate indices for each language:

```bash
# Index your English documents
python -m src.embedder \
  --model-dir ./fine_tuned_splade_coldchain_en \
  --documents-file /path/to/your/english_documents.json \
  --output-file ./coldchain_document_vectors_en.npz

# Index your Dutch documents
python -m src.embedder \
  --model-dir ./fine_tuned_splade_coldchain_nl \
  --documents-file /path/to/your/dutch_documents.json \
  --output-file ./coldchain_document_vectors_nl.npz
```

### Approach 2: Language Detection in Your Application

In your application, detect the language of the user's query and route it to the appropriate model:

```python
# Example pseudo-code for language routing
def search(query, documents):
    # Simple language detection (in production, use a proper language detection library)
    if any(word in query.lower() for word in ['het', 'de', 'een', 'voor']):
        # Dutch query
        model_dir = './fine_tuned_splade_coldchain_nl'
    else:
        # Default to English
        model_dir = './fine_tuned_splade_coldchain_en'
    
    # Load appropriate model and perform search
    tokenizer, model, device = load_splade_model(model_dir)
    results = retrieve_documents(query, documents, tokenizer, model, device)
    return results
```

## Performance Considerations

### Hardware Requirements

While Ollama can run on modest hardware, generation will be much faster with:
- A dedicated GPU (at least 8GB VRAM)
- 32GB+ system RAM
- Multi-core CPU

On a system with an NVIDIA RTX 3080, the entire process (data generation + training for both languages) can complete in about 6-8 hours. On CPU-only systems, expect 20-30 hours total.

### Optimizing Generation

If generation is slow, try:
- Reducing the number of queries (start with 50 instead of 100)
- Using a smaller model like `phi3:mini` instead of `qwen2.5`
- Increasing the number of workers if you have a multi-core CPU

## Advantages of Local Generation with Ollama

Using Ollama for local generation offers several benefits:

1. **Privacy**: Your domain-specific data never leaves your system
2. **Cost**: No API usage fees or rate limits
3. **Customization**: Full control over the generation process
4. **Reproducibility**: Consistent results with fixed model versions

## Limitations and Considerations

- Local generation will be slower than using commercial APIs, especially without a GPU
- Dutch results may require more validation than English ones
- Each language requires its own separate model and training process
- Generated training data should still be validated for critical applications

## Conclusion

By combining Ollama's local LLM capabilities with Domain Distiller and the SPLADE Model Trainer, you've created a completely self-contained pipeline for building multilingual, domain-specific search systems. This approach gives you powerful neural search capabilities for cold-chain logistics without the traditional barriers of data collection or expensive API calls.

The language-specific approach used by Domain Distiller—running separate processes for each language—ensures that each model is optimized for its target language. This results in better search quality compared to a one-size-fits-all multilingual model.

Whether you're building a quality management system, a documentation portal, or a knowledge base for cold-chain logistics, this approach offers a practical and cost-effective path to advanced multilingual search capabilities.
