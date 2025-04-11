# Introduction to SPLADE: Sparse Lexical and Expansion Models for Information Retrieval

**Author:** SPLADE Team  
**Date:** 2025-04-11

## Introduction

Information retrieval (IR) has seen a significant transformation in recent years with the advent of neural models. Traditional keyword-based search approaches have been augmented or replaced by dense vector models like BERT and other Transformer-based architectures. However, these dense retrieval approaches, while effective, can be computationally expensive and lack explainability.

This is where SPLADE (SParse Lexical AnD Expansion) models come in. SPLADE models offer a compelling middle ground, combining the efficiency of sparse retrievers with the effectiveness of neural language models.

## What is SPLADE?

SPLADE is a neural retrieval model that uses sparse representations for both queries and documents. Unlike dense retrieval models that encode everything into continuous vector spaces, SPLADE creates sparse representations over the vocabulary space. Each dimension of the sparse vector corresponds to a specific word or token in the vocabulary.

The key insight of SPLADE is that it learns to perform **lexical expansion** - the model doesn't just activate the exact words present in the text, but also semantically related terms. This expansion is learned during training and allows for more robust matching between queries and documents.

## How SPLADE Works

At a high level, SPLADE works as follows:

1. The model takes an input text (either a query or a document)
2. It passes this text through a Transformer-based model (typically BERT)
3. For each token position, it produces a weight for every word in the vocabulary
4. These weights are transformed using a ReLU activation followed by a log(1+x) transformation
5. Max-pooling is applied across all token positions to get a single sparse vector

The resulting sparse vector has non-zero values for the words in the original text, as well as for semantically related terms that were "expanded" by the model. This sparse representation can then be efficiently stored and retrieved using traditional inverted index technologies.

## Advantages of SPLADE

SPLADE offers several advantages over both traditional keyword search and dense retrieval models:

- **Efficiency**: Sparse representations can leverage efficient inverted index data structures
- **Explainability**: Each dimension corresponds to a word, making it easier to understand why a document was retrieved
- **Expansion capability**: The model learns to expand queries and documents with related terms, improving recall
- **Compatibility**: Works with existing information retrieval infrastructure (e.g., Elasticsearch, Lucene)

## SPLADE vs. Other Approaches

Here's how SPLADE compares to other retrieval approaches:

| Approach | Vector Representation | Expansion | Inference Speed | Index Size | Explainability |
|----------|------------------------|-----------|-----------------|------------|----------------|
| BM25 | Sparse | No | Fast | Small | High |
| Dense Retrieval (e.g., DPR) | Dense | Implicit | Medium | Large | Low |
| SPLADE | Sparse | Yes | Medium | Medium | Medium-High |

## Training SPLADE Models

Training a SPLADE model typically involves the following steps:

1. Starting with a pre-trained language model (e.g., BERT)
2. Fine-tuning on query-document pairs using contrastive learning
3. Applying regularization to control the sparsity level

The SPLADE Model Trainer toolkit in this repository provides all the necessary tools to train, evaluate, and deploy SPLADE models on your own data.

## Code Example: Using SPLADE for Document Retrieval

Here's a simple example of how to use a trained SPLADE model for document retrieval:

```python
from src.embedder import SpladeEmbedder
from src.tests.code.test_queries import retrieve_documents

# Initialize the embedder
embedder = SpladeEmbedder(model_dir="./fine_tuned_splade")

# Prepare document collection
documents = [
    "SPLADE combines the efficiency of sparse retrievers with neural expansion.",
    "Dense retrieval models encode queries and documents in continuous vector spaces.",
    "Traditional search engines use inverted indices for fast keyword lookup.",
    "Neural information retrieval has advanced significantly in recent years."
]

# Encode documents
doc_embeddings = [embedder.encode_text(doc) for doc in documents]

# Process a query
query = "How do neural search systems work?"
query_embedding = embedder.encode_text(query)

# Compute scores and rank documents
scores = [sum(q_val * d_val for q_val, d_val in zip(
    query_embedding['values'], doc_embedding['values']))
    for doc_embedding in doc_embeddings]

# Get top results
ranked_docs = [(score, doc) for score, doc in sorted(
    zip(scores, documents), reverse=True)]

# Print results
for score, doc in ranked_docs:
    print(f"Score: {score:.4f}, Document: {doc}")
```

## Future Directions

The field of sparse neural retrieval is rapidly evolving. Some promising directions include:

- **Hybrid approaches**: Combining SPLADE with dense retrievers for even better performance
- **Efficiency improvements**: Further optimizing the sparsity pattern for faster inference
- **Multi-lingual SPLADE**: Extending the approach to support multiple languages
- **Domain adaptation**: Techniques for adapting SPLADE models to specific domains with minimal data

## Conclusion

SPLADE represents an important advancement in neural information retrieval, offering a compelling balance between the efficiency of traditional search engines and the effectiveness of neural models. With the SPLADE Model Trainer toolkit, you can easily train and deploy these models for your own applications.

Whether you're building a document search system, a question-answering application, or any other IR system, SPLADE is worth considering as a powerful and efficient retrieval approach.

## References

- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720) (Formal & Lassance et al., 2021)
- [SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval](https://arxiv.org/abs/2109.10086) (Formal et al., 2021)
- [Efficient Passage Retrieval with Hashing for Open-domain Question Answering](https://arxiv.org/abs/2106.00882) (Yang et al., 2021)
- [From doc2query to docTTTTTquery](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf) (Nogueira & Lin, 2019)
