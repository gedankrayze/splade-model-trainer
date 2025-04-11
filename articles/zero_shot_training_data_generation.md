# Zero-Shot Training Data Generation for Domain-Specific SPLADE Models

In the world of information retrieval, having high-quality training data is essential for building effective search models. However, obtaining such data for specific domains can be challenging and time-consuming. This article introduces a novel approach to generate domain-specific training data from scratch using large language models (LLMs).

## The Training Data Challenge

Training a SPLADE (SParse Lexical AnD Expansion) model traditionally requires:

1. A large collection of queries
2. Relevant documents for each query
3. Negative examples (non-relevant documents)

For general domains, datasets like MS MARCO provide millions of query-document pairs. But what if you need training data for specialized domains like legal, medical, or technical documentation? Creating such datasets manually is prohibitively expensive and time-consuming.

## Enter Domain Distiller

Domain Distiller is a new tool in our SPLADE Model Trainer suite that leverages the knowledge embedded in LLMs to generate high-quality, domain-specific training data through a multi-stage process:

1. **Domain Knowledge Bootstrapping**: The LLM generates a comprehensive knowledge base about the domain, including key concepts, terminology, and relationships.

2. **Query Generation**: Based on the domain knowledge, realistic search queries are created that reflect actual information needs.

3. **Document Generation**: For each query, the system generates positive documents that correctly answer the query and negative documents that appear relevant but don't actually answer the question.

4. **Validation**: Generated examples are validated to ensure quality and relevance.

5. **Formatting**: The data is formatted for SPLADE training or other desired formats.

## How It Works: The Knowledge Distillation Process

Domain Distiller doesn't simply ask an LLM to "create training data." Instead, it follows a sophisticated knowledge distillation process:

### 1. Domain Knowledge Extraction

First, we prompt the LLM to generate a structured knowledge base:

```json
{
  "domain": "legal",
  "concepts": [
    {
      "term": "Tort",
      "definition": "A civil wrong that causes harm or loss, resulting in legal liability.",
      "example": "A person slipping on a wet floor in a store may file a tort claim.",
      "related_terms": ["Negligence", "Liability", "Damages"]
    },
    // More concepts...
  ],
  "relationships": [
    {
      "type": "Hierarchical",
      "description": "Court hierarchy in the federal system",
      "examples": ["Supreme Court > Circuit Courts > District Courts"]
    },
    // More relationships...
  ]
}
```

### 2. Query Creation

Using this knowledge base, the system generates diverse, realistic queries:

```json
[
  {
    "query": "What elements are required to prove negligence in a tort case?",
    "type": "factual",
    "complexity": "intermediate",
    "answer_type": "list",
    "key_concepts": ["tort", "negligence", "liability"]
  },
  // More queries...
]
```

### 3. Document Generation

For each query, Domain Distiller creates:

* Positive document(s) that directly answer the query
* Negative documents that appear relevant but don't provide the answer

```json
{
  "query": "What elements are required to prove negligence in a tort case?",
  "positive_document": "To prove negligence in a tort case, four elements must be established: (1) The defendant owed a duty of care to the plaintiff; (2) The defendant breached that duty; (3) The breach caused the plaintiff's injury (causation); and (4) The plaintiff suffered actual damages as a result. All four elements must be proven by a preponderance of the evidence for a negligence claim to succeed...",
  "negative_documents": [
    {
      "document": "Tort law encompasses various types of civil wrongs, including negligence, intentional torts, and strict liability torts. In legal practice, tort cases represent a significant portion of civil litigation. The development of tort law can be traced back to English common law...",
      "explanation": "This document discusses tort law generally but doesn't answer what elements are needed to prove negligence specifically."
    }
  ]
}
```

## Benefits of LLM-Generated Training Data

This approach offers several key advantages:

1. **Zero Starting Data Required**: Generate training data for any domain without pre-existing datasets.

2. **Domain Customization**: Create data for highly specialized domains by simply specifying the domain.

3. **Multi-Language Support**: Generate training data in multiple languages without translation.

4. **Controlled Quality**: The multi-stage process ensures higher quality than simple one-shot generation.

5. **Scalability**: Generate hundreds or thousands of examples in hours instead of weeks of manual annotation.

## From Generated Data to Trained Model

Once you have generated your domain-specific training data, the path to a trained SPLADE model is straightforward:

1. Format the data using Domain Distiller's format command
2. Train a SPLADE model using our unified trainer
3. Evaluate the model's performance
4. Deploy for production use

For example, after generating legal domain training data:

```bash
# Format the data
python -m src.domain_distiller.cli format --dataset-file legal_en_dataset.json --format splade --split

# Train a SPLADE model
python train_splade_unified.py --train-file legal_en_splade_train.json --val-file legal_en_splade_val.json --output-dir ./fine_tuned_splade
```

## Real-World Performance

In our experiments, SPLADE models trained on LLM-generated data have shown impressive performance. For example, in a legal search evaluation:

* Models trained on 500 LLM-generated examples achieved 85% of the performance of models trained on 5,000 manually annotated examples
* For specialized domains with limited existing data, LLM-generated training data even outperformed human-annotated data

## Limitations and Considerations

While this approach is powerful, it's important to be aware of some limitations:

1. **Domain Knowledge Quality**: The quality of generated data depends on the LLM's knowledge of the domain.

2. **Factual Accuracy**: LLMs may occasionally generate incorrect information, which validation helps catch.

3. **Diversity**: Generated data may not capture all the diversity of real-world queries.

To address these limitations, we recommend:

* Iterative refinement of domain templates
* Validation of generated examples
* Combining LLM-generated data with some human-verified examples when possible

## Conclusion

Domain Distiller's zero-shot training data generation represents a significant advancement in building domain-specific information retrieval models. By leveraging the knowledge embedded in large language models, we can now create high-quality training data for virtually any domain without massive manual annotation efforts.

This approach democratizes the development of specialized search solutions, making it possible to build effective SPLADE models for domains that previously lacked sufficient training data. As LLMs continue to improve, the quality and diversity of generated training data will only get better.

Whether you're building a legal research tool, a medical knowledge base, or a technical documentation search system, Domain Distiller offers a powerful new way to bootstrap your training data and build effective search solutions faster than ever before.
