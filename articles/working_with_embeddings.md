# Working with Embeddings: Dense Vectors, Cosine Similarity, and Hybrid Search

This article covers the essential concepts and practical techniques for working with embeddings in the SPLADE Model Trainer toolkit. While our project primarily focuses on SPLADE sparse representations, we also provide complementary support for dense embeddings and hybrid approaches that combine both paradigms.

## Introduction to Embeddings

Embeddings are numerical representations of text that capture semantic meaning in a way that computers can process. In information retrieval, two main types of embeddings are commonly used:

1. **Dense Embeddings**: Fixed-size continuous vectors where meaning is distributed across all dimensions
2. **Sparse Embeddings (like SPLADE)**: Sparse vectors where specific dimensions correspond to specific vocabulary terms

Each approach has strengths and weaknesses:
- Dense embeddings excel at capturing semantic relationships and similarity
- Sparse embeddings (like SPLADE) provide better efficiency, explainability, and lexical matching

## Working with Dense Embeddings

### Generating Document Embeddings

To embed documents using our toolkit:

```python
from src.embedder import TextEmbedder

# Initialize embedder with a specific model
embedder = TextEmbedder(model_name="BAAI/bge-small-en-v1.5")

# Generate embeddings for a batch of documents
documents = [
    "This document discusses machine learning applications.",
    "Information retrieval systems help users find relevant content.",
    "Neural networks have transformed natural language processing."
]

# Get normalized document embeddings (list of vectors)
document_embeddings = embedder.generate_embeddings(documents)

# document_embeddings is a list of lists, where each inner list is a vector
# Example: [[-0.021, 0.084, -0.012, ...], [0.056, -0.032, 0.087, ...], ...]
```

### Generating Query Embeddings

Generating query embeddings follows the same pattern:

```python
# For a single query
query = "How do machine learning systems improve search?"
query_embedding = embedder.generate_embeddings(query)[0]  # Get the first (and only) embedding
```

**Important**: When using models like E5, BAAI/BGE, or GTR, you should add a prefix to your queries for best results:

```python
# E5 models use "query: " prefix
query = "query: " + "How do machine learning systems improve search?"

# BGE models use "query: " prefix 
query = "query: " + "How do machine learning systems improve search?"

# For document embeddings with these models, prefixes may also help:
documents = ["passage: " + doc for doc in documents]  # For E5/BGE models
```

Check your specific model's documentation for recommended prefixes.

### Computing Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors, providing a similarity score between -1 and 1 (though with normalized embeddings, the range is typically 0 to 1).

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity between a query and multiple documents
similarities = cosine_similarity(
    [query_embedding],      # 2D array with a single query embedding
    document_embeddings     # 2D array of document embeddings
)[0]  # Take first row from the result to get a 1D array

# Sort documents by similarity score (descending)
ranked_indices = np.argsort(similarities)[::-1]
for i, idx in enumerate(ranked_indices):
    print(f"Rank {i+1}: Document '{documents[idx][:50]}...' (Score: {similarities[idx]:.4f})")
```

If you don't want to use scikit-learn, here's a simple NumPy implementation:

```python
def cosine_sim(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector (can be list or numpy array)
        vec2: Second vector (can be list or numpy array)
        
    Returns:
        Cosine similarity as float between -1 and 1
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Handle zero-length vectors to avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return dot_product / (norm1 * norm2)
```

## Building a Simple Vector Search System

Here's a basic end-to-end example of a vector search system:

```python
import numpy as np
from src.embedder import TextEmbedder

class SimpleVectorSearch:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.embedder = TextEmbedder(model_name=model_name)
        self.documents = []
        self.document_embeddings = []
        
    def add_documents(self, documents):
        """Add documents to the search index."""
        self.documents.extend(documents)
        
        # Generate embeddings for new documents
        new_embeddings = self.embedder.generate_embeddings(documents)
        self.document_embeddings.extend(new_embeddings)
        
    def search(self, query, top_k=5):
        """Search for documents most similar to the query."""
        # Add appropriate prefix based on model type
        query = "query: " + query  # For BAAI/BGE and E5 models
        
        # Generate query embedding
        query_embedding = self.embedder.generate_embeddings(query)[0]
        
        # Compute similarities
        similarities = []
        for doc_embedding in self.document_embeddings:
            similarity = np.dot(query_embedding, doc_embedding)  # Simplified dot product
            similarities.append(similarity)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                "rank": i + 1,
                "document": self.documents[idx],
                "score": similarities[idx]
            })
            
        return results

# Usage:
search_engine = SimpleVectorSearch()
search_engine.add_documents([
    "SPLADE combines sparse and lexical properties for efficient search.",
    "Dense retrievers encode semantic meaning in continuous vectors.",
    "Transformer models have revolutionized natural language processing.",
    "The hybrid approach combines multiple retrieval strategies."
])

results = search_engine.search("How do vector search systems work?")
for result in results:
    print(f"Rank {result['rank']}: {result['document']} (Score: {result['score']:.4f})")
```

## Using Our Hybrid Embedder

The toolkit provides a `HybridEmbedder` that can generate both dense and sparse embeddings:

```python
from src.embedder import HybridEmbedder

# Initialize with both dense and SPLADE models
embedder = HybridEmbedder(
    dense_model_name="BAAI/bge-small-en-v1.5",
    splade_model_dir="./fine_tuned_splade/splade-model"  # Path to your fine-tuned SPLADE model
)

# Generate both types of embeddings at once
text = "Hybrid search combines multiple retrieval paradigms."
hybrid_embedding = embedder.generate_hybrid_embeddings(text)[0]

# Access individual components
dense_vector = hybrid_embedding['dense']  # A list of floats
sparse_vector = hybrid_embedding['sparse']  # Dict with 'indices' and 'values' lists

print(f"Dense embedding has {len(dense_vector)} dimensions")
print(f"Sparse embedding has {len(sparse_vector['indices'])} non-zero entries")
```

## Advanced: Quantization and Efficiency

For production systems with large document collections, consider these optimizations:

### Vector Quantization

Quantization reduces precision to save memory:

```python
# Example using 8-bit quantization with numpy
import numpy as np

def quantize_to_int8(vectors):
    """Quantize float32 vectors to int8 for efficiency."""
    # Find min and max to scale appropriately
    max_val = np.max(vectors)
    min_val = np.min(vectors)
    
    # Scale to int8 range (-128 to 127)
    scale = 255.0 / (max_val - min_val) if max_val != min_val else 1.0
    zero_point = -min_val * scale - 128
    
    # Quantize
    quantized = np.clip(np.round(vectors * scale + zero_point), -128, 127).astype(np.int8)
    
    # Store scale and zero_point to dequantize later
    metadata = {
        'scale': scale,
        'zero_point': zero_point
    }
    
    return quantized, metadata

def dequantize_from_int8(quantized, metadata):
    """Convert quantized vectors back to float."""
    scale = metadata['scale']
    zero_point = metadata['zero_point']
    
    # Dequantize
    dequantized = (quantized.astype(np.float32) - zero_point) / scale
    
    return dequantized
```

### Efficient Similarity Search

For large collections, consider:

1. **Approximate Nearest Neighbor (ANN)** libraries:
   - Faiss (Facebook AI Similarity Search)
   - Annoy (Spotify)
   - ScaNN (Google)
   - HNSW (Hierarchical Navigable Small World)

2. **Vector Databases**:
   - Qdrant
   - Milvus
   - Weaviate
   - Pinecone

Example with Faiss:

```python
import faiss
import numpy as np

# Convert embeddings to float32 numpy array
embeddings = np.array(document_embeddings).astype(np.float32)
dimension = embeddings.shape[1]

# Create an index
index = faiss.IndexFlatIP(dimension)  # Inner product (for normalized vectors)
index.add(embeddings)

# Search
k = 5  # number of nearest neighbors
query_vector = np.array([query_embedding]).astype(np.float32)
distances, indices = index.search(query_vector, k)

# Process results
for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"Rank {i+1}: Document '{documents[idx][:50]}...' (Score: {distance:.4f})")
```

## Hybrid Retrieval Strategies

Hybrid retrieval combines sparse and dense approaches. Our toolkit supports:

1. **Separate Retrievers**: Run SPLADE and dense embedding retrievers separately, then merge results

```python
def hybrid_search(query, documents, splade_embedder, dense_embedder, alpha=0.5, top_k=5):
    """
    Perform hybrid search using both SPLADE and dense embeddings.
    
    Args:
        query: The search query
        documents: List of documents to search
        splade_embedder: SPLADE embedder instance
        dense_embedder: Dense embedder instance
        alpha: Weight for SPLADE scores (1-alpha applied to dense scores)
        top_k: Number of results to return
        
    Returns:
        List of top-k documents with scores
    """
    # Get SPLADE embeddings
    query_splade = splade_embedder.encode_text(query)
    doc_splade_embeddings = [splade_embedder.encode_text(doc) for doc in documents]
    
    # Get dense embeddings
    query_dense = dense_embedder.generate_embeddings("query: " + query)[0]
    doc_dense_embeddings = dense_embedder.generate_embeddings(["passage: " + doc for doc in documents])
    
    # Compute scores
    hybrid_scores = []
    
    for i, doc in enumerate(documents):
        # SPLADE score (dot product of sparse vectors)
        splade_score = 0
        for idx, val in zip(query_splade['indices'], query_splade['values']):
            if idx in doc_splade_embeddings[i]['indices']:
                doc_idx = doc_splade_embeddings[i]['indices'].index(idx)
                splade_score += val * doc_splade_embeddings[i]['values'][doc_idx]
        
        # Dense score (cosine similarity)
        dense_score = np.dot(query_dense, doc_dense_embeddings[i])
        
        # Combine scores
        hybrid_score = alpha * splade_score + (1 - alpha) * dense_score
        hybrid_scores.append((i, hybrid_score))
    
    # Sort by score (descending)
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k results
    results = []
    for i, (idx, score) in enumerate(hybrid_scores[:top_k]):
        results.append({
            "rank": i + 1,
            "document": documents[idx],
            "score": score
        })
    
    return results
```

2. **Unified Embedder**: Use our HybridEmbedder for both types at once:

```python
from src.embedder import HybridEmbedder

# Initialize hybrid embedder
hybrid_embedder = HybridEmbedder(
    dense_model_name="BAAI/bge-small-en-v1.5", 
    splade_model_dir="./fine_tuned_splade/splade-model"
)

# Generate hybrid embeddings
hybrid_docs = hybrid_embedder.generate_hybrid_embeddings(documents)
hybrid_query = hybrid_embedder.generate_hybrid_embeddings(query)[0]

# Custom hybrid scoring function
def hybrid_score(query_hybrid, doc_hybrid, alpha=0.5):
    # Sparse component (dot product)
    sparse_score = 0
    q_indices = set(query_hybrid['sparse']['indices'])
    d_indices = set(doc_hybrid['sparse']['indices'])
    common_indices = q_indices.intersection(d_indices)
    
    for idx in common_indices:
        q_idx = query_hybrid['sparse']['indices'].index(idx)
        d_idx = doc_hybrid['sparse']['indices'].index(idx)
        sparse_score += query_hybrid['sparse']['values'][q_idx] * doc_hybrid['sparse']['values'][d_idx]
    
    # Dense component (dot product of normalized vectors)
    dense_score = np.dot(query_hybrid['dense'], doc_hybrid['dense'])
    
    # Combine with weighted interpolation
    return alpha * sparse_score + (1 - alpha) * dense_score
```

## Conclusion

While our toolkit focuses primarily on SPLADE for information retrieval, these complementary embedding functionalities provide flexibility for diverse search requirements. Dense embeddings can effectively capture semantic relationships, while SPLADE excels at efficiency and explainability. For many applications, a hybrid approach offers the best of both worlds.

To learn more about our main focus on SPLADE models, see the other articles in this series:
- [Introduction to SPLADE](introduction_to_splade.md)
- [Unified Trainer](introducing_unified_trainer.md)
- [Domain Distiller for Training Data Generation](using_ollama_with_domain_distiller.md)

## Further Reading

- [Understanding Dense Vector Embeddings](https://huggingface.co/blog/dense-embeddings)
- [Approximate Nearest Neighbor Search for Vector Databases](https://www.pinecone.io/learn/vector-indexes/)
- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)
- [Hybrid Retrieval Approaches in Information Retrieval](https://arxiv.org/abs/2012.08574)