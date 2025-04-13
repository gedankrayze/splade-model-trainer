# Scaling Search with FAISS: Implementing SPLADE, Dense, and Hybrid Search

This article demonstrates how to implement efficient search at scale using FAISS (Facebook AI Similarity Search) with our SPLADE Model Trainer toolkit. We'll cover three search approaches:

1. Classic dense vector search with FAISS
2. SPLADE sparse vector search with FAISS
3. Hybrid search combining both approaches

## Introduction to FAISS

FAISS is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, even ones that don't fit in RAM. Its key advantages include:

- High performance with GPU support
- Scalability to billions of vectors
- Support for different indexing methods with speed/accuracy tradeoffs
- Approximate nearest neighbor algorithms for faster search

## Setup and Installation

First, install FAISS along with our toolkit requirements:

```bash
# CPU-only version
pip install faiss-cpu

# Or GPU version (requires CUDA)
pip install faiss-gpu
```

Let's also import the necessary libraries for our examples:

```python
import numpy as np
import faiss
import json
import time
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

# Import our toolkit components
from src.embedder import TextEmbedder, SpladeEmbedder, HybridEmbedder
```

## 1. Dense Vector Search with FAISS

Dense embeddings map text to continuous vectors where semantic meaning is distributed across all dimensions. Let's implement a search system using dense embeddings with FAISS:

```python
class DenseVectorSearch:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", use_gpu=False):
        """
        Initialize dense vector search with FAISS.
        
        Args:
            model_name: Name of the dense embedding model
            use_gpu: Whether to use GPU for FAISS (if available)
        """
        # Initialize the embedder
        self.embedder = TextEmbedder(model_name=model_name)
        self.use_gpu = use_gpu
        self.index = None
        self.documents = []
        self.dimension = None
        self.index_built = False
    
    def add_documents(self, documents: List[str], batch_size=32):
        """
        Add documents to the search index.
        
        Args:
            documents: List of documents to add
            batch_size: Batch size for processing documents
        """
        self.documents.extend(documents)
        
        # Process documents in batches to avoid memory issues
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            # Add passage prefix for better retrieval performance
            batch = ["passage: " + doc for doc in batch]
            batch_embeddings = self.embedder.generate_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
        
        # Convert to numpy array
        embeddings = np.array(all_embeddings).astype(np.float32)
        
        # Create or update index
        if self.index is None:
            self.dimension = embeddings.shape[1]
            self._create_index(embeddings)
        else:
            self.index.add(embeddings)
    
    def _create_index(self, initial_vectors):
        """
        Create FAISS index for the vectors.
        
        Args:
            initial_vectors: Initial vectors to add to the index
        """
        dimension = initial_vectors.shape[1]
        self.dimension = dimension
        
        # For smaller collections, use exact search with L2 distance
        if len(self.documents) < 10000:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity (normalized vectors)
        else:
            # For larger collections, use IVF index for faster search
            # The nlist parameter controls the number of Voronoi cells
            nlist = min(4096, max(int(len(self.documents) / 39), 256))
            
            # Define quantizer
            quantizer = faiss.IndexFlatIP(dimension)
            
            # Create IVF index
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Need to train the index
            self.index.train(initial_vectors)
        
        # Add vectors to the index
        self.index.add(initial_vectors)
        
        # Use GPU if available and requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print(f"Moving index to GPU...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.index_built = True
    
    def search(self, query: str, top_k=10, nprobe=None):
        """
        Search for documents similar to the query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            nprobe: Number of cells to visit for IVF indexes (higher = more accurate but slower)
        
        Returns:
            List of dictionaries with document and score
        """
        if not self.index_built:
            raise ValueError("Index has not been built. Add documents first.")
        
        # Add query prefix for better retrieval
        query = "query: " + query
        
        # Generate query embedding
        query_embedding = np.array(self.embedder.generate_embeddings(query)[0]).astype(np.float32).reshape(1, -1)
        
        # Set nprobe for IVF indexes (more cells = more accurate but slower)
        if isinstance(self.index, faiss.IndexIVFFlat) and nprobe is not None:
            self.index.nprobe = nprobe
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Prepare results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]
            if idx != -1:  # Skip invalid indices
                results.append({
                    "document": self.documents[idx],
                    "score": float(distance),
                    "id": int(idx)
                })
        
        return results
    
    def save(self, filepath: str):
        """
        Save the index and documents to disk.
        
        Args:
            filepath: Base filepath to save index and documents
        """
        # Save the index
        index_path = f"{filepath}.index"
        
        # If using GPU, move index back to CPU for saving
        if self.use_gpu and faiss.get_num_gpus() > 0:
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        else:
            index_to_save = self.index
            
        faiss.write_index(index_to_save, index_path)
        
        # Save documents and metadata
        data = {
            "documents": self.documents,
            "dimension": self.dimension,
            "model_name": self.embedder.model_name
        }
        
        with open(f"{filepath}.json", "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str, use_gpu=False):
        """
        Load index and documents from disk.
        
        Args:
            filepath: Base filepath to load index and documents from
            use_gpu: Whether to use GPU for search
            
        Returns:
            DenseVectorSearch instance
        """
        # Load documents and metadata
        with open(f"{filepath}.json", "r") as f:
            data = json.load(f)
        
        # Create instance with the same model
        instance = cls(model_name=data["model_name"], use_gpu=use_gpu)
        instance.documents = data["documents"]
        instance.dimension = data["dimension"]
        
        # Load the index
        instance.index = faiss.read_index(f"{filepath}.index")
        
        # Move index to GPU if requested
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            instance.index = faiss.index_cpu_to_gpu(res, 0, instance.index)
        
        instance.index_built = True
        
        return instance


# Example usage:
if __name__ == "__main__":
    # Create search instance
    dense_search = DenseVectorSearch(model_name="BAAI/bge-small-en-v1.5")
    
    # Add documents
    documents = [
        "SPLADE is a sparse retrieval model that combines lexical and semantic matching.",
        "Dense retrieval models map queries and documents to continuous vector spaces.",
        "Hybrid search combines multiple retrieval methods for better results.",
        "FAISS is a library for efficient similarity search developed by Facebook AI Research."
    ]
    
    dense_search.add_documents(documents)
    
    # Search
    results = dense_search.search("How do vector search systems work?")
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result['document']} (Score: {result['score']:.4f})")
    
    # Save and load (optional)
    dense_search.save("./my_dense_index")
    loaded_search = DenseVectorSearch.load("./my_dense_index")
```

## 2. SPLADE Sparse Vector Search with FAISS

SPLADE generates sparse vectors where each dimension corresponds to a term in the vocabulary. While FAISS is designed primarily for dense vectors, we can adapt it for sparse vectors:

```python
class SpladeSearch:
    def __init__(self, splade_model_dir, use_gpu=False):
        """
        Initialize SPLADE search with FAISS.
        
        Args:
            splade_model_dir: Directory containing SPLADE model
            use_gpu: Whether to use GPU for FAISS (if available)
        """
        # Initialize the SPLADE embedder
        self.embedder = SpladeEmbedder(model_dir=splade_model_dir)
        self.use_gpu = use_gpu
        self.index = None
        self.documents = []
        self.vocabulary_size = None  # Will be determined from first embedding
        self.index_built = False
    
    def add_documents(self, documents: List[str], batch_size=16):
        """
        Add documents to the search index.
        
        Args:
            documents: List of documents to add
            batch_size: Batch size for processing documents
        """
        self.documents.extend(documents)
        
        # Process documents in batches to avoid memory issues
        all_sparse_vectors = []
        for i in tqdm(range(0, len(documents), batch_size), desc="Encoding documents"):
            batch = documents[i:i+batch_size]
            batch_sparse = self.embedder.generate_sparse_embeddings(batch)
            all_sparse_vectors.extend(batch_sparse)
        
        # Convert sparse representations to dense for FAISS
        # For the first batch, determine vocabulary size
        if self.vocabulary_size is None and all_sparse_vectors:
            # Get vocabulary size from tokenizer
            self.vocabulary_size = len(self.embedder.tokenizer)
            print(f"Vocabulary size: {self.vocabulary_size}")
        
        # Convert sparse to dense vectors
        dense_vectors = self._sparse_to_dense(all_sparse_vectors)
        
        # Create or update index
        if self.index is None:
            self._create_index(dense_vectors)
        else:
            self.index.add(dense_vectors)
    
    def _sparse_to_dense(self, sparse_vectors):
        """
        Convert sparse vectors to dense format for FAISS.
        
        Args:
            sparse_vectors: List of sparse vectors
            
        Returns:
            Dense numpy array
        """
        dense_vectors = np.zeros((len(sparse_vectors), self.vocabulary_size), dtype=np.float32)
        
        for i, sparse_vec in enumerate(sparse_vectors):
            indices = sparse_vec['indices']
            values = sparse_vec['values']
            
            for idx, val in zip(indices, values):
                if idx < self.vocabulary_size:
                    dense_vectors[i, idx] = val
        
        return dense_vectors
    
    def _create_index(self, initial_vectors):
        """
        Create FAISS index for the vectors.
        
        Args:
            initial_vectors: Initial vectors to add to the index
        """
        # For SPLADE vectors, we use inner product similarity
        self.index = faiss.IndexFlatIP(self.vocabulary_size)
        
        # Add vectors to the index
        self.index.add(initial_vectors)
        
        # Use GPU if available and requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print(f"Moving index to GPU...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.index_built = True
    
    def search(self, query: str, top_k=10):
        """
        Search for documents similar to the query.
        
        Args:
            query: Query text
            top_k: Number of results to return
        
        Returns:
            List of dictionaries with document and score
        """
        if not self.index_built:
            raise ValueError("Index has not been built. Add documents first.")
        
        # Generate query embedding (sparse)
        query_sparse = self.embedder.encode_text(query)
        
        # Convert to dense format for FAISS
        query_dense = np.zeros((1, self.vocabulary_size), dtype=np.float32)
        for idx, val in zip(query_sparse['indices'], query_sparse['values']):
            if idx < self.vocabulary_size:
                query_dense[0, idx] = val
        
        # Search
        distances, indices = self.index.search(query_dense, min(top_k, len(self.documents)))
        
        # Prepare results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]
            if idx != -1:  # Skip invalid indices
                results.append({
                    "document": self.documents[idx],
                    "score": float(distance),
                    "id": int(idx)
                })
        
        return results
    
    def save(self, filepath: str):
        """
        Save the index and documents to disk.
        
        Args:
            filepath: Base filepath to save index and documents
        """
        # Save the index
        index_path = f"{filepath}.index"
        
        # If using GPU, move index back to CPU for saving
        if self.use_gpu and faiss.get_num_gpus() > 0:
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        else:
            index_to_save = self.index
            
        faiss.write_index(index_to_save, index_path)
        
        # Save documents and metadata
        data = {
            "documents": self.documents,
            "vocabulary_size": self.vocabulary_size,
            "splade_model_dir": self.embedder.model_dir
        }
        
        with open(f"{filepath}.json", "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str, use_gpu=False):
        """
        Load index and documents from disk.
        
        Args:
            filepath: Base filepath to load index and documents from
            use_gpu: Whether to use GPU for search
            
        Returns:
            SpladeSearch instance
        """
        # Load documents and metadata
        with open(f"{filepath}.json", "r") as f:
            data = json.load(f)
        
        # Create instance with the same model
        instance = cls(splade_model_dir=data["splade_model_dir"], use_gpu=use_gpu)
        instance.documents = data["documents"]
        instance.vocabulary_size = data["vocabulary_size"]
        
        # Load the index
        instance.index = faiss.read_index(f"{filepath}.index")
        
        # Move index to GPU if requested
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            instance.index = faiss.index_cpu_to_gpu(res, 0, instance.index)
        
        instance.index_built = True
        
        return instance


# Example usage:
if __name__ == "__main__":
    # Create search instance
    splade_search = SpladeSearch(splade_model_dir="./fine_tuned_splade/splade-model")
    
    # Add documents
    documents = [
        "SPLADE is a sparse retrieval model that combines lexical and semantic matching.",
        "Dense retrieval models map queries and documents to continuous vector spaces.",
        "Hybrid search combines multiple retrieval methods for better results.",
        "FAISS is a library for efficient similarity search developed by Facebook AI Research."
    ]
    
    splade_search.add_documents(documents)
    
    # Search
    results = splade_search.search("How do retrieval models work?")
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result['document']} (Score: {result['score']:.4f})")
```

## 3. Hybrid Search with FAISS

Hybrid search combines the strengths of both dense and sparse retrieval. Let's implement a hybrid approach:

```python
class HybridSearch:
    def __init__(self, dense_model_name="BAAI/bge-small-en-v1.5",
                 splade_model_dir="./fine_tuned_splade/splade-model", 
                 use_gpu=False,
                 alpha=0.5):  # Weight factor for combining scores
        """
        Initialize hybrid search with both dense and SPLADE models.
        
        Args:
            dense_model_name: Name of dense embedding model
            splade_model_dir: Directory containing SPLADE model
            use_gpu: Whether to use GPU for FAISS
            alpha: Weight factor for combining scores (alpha * sparse + (1-alpha) * dense)
        """
        # Initialize both search systems
        self.dense_search = DenseVectorSearch(model_name=dense_model_name, use_gpu=use_gpu)
        self.splade_search = SpladeSearch(splade_model_dir=splade_model_dir, use_gpu=use_gpu)
        self.documents = []
        self.alpha = alpha
    
    def add_documents(self, documents: List[str], batch_size=16):
        """
        Add documents to both indexes.
        
        Args:
            documents: List of documents to add
            batch_size: Batch size for processing
        """
        self.documents.extend(documents)
        
        # Add to both indexes
        self.dense_search.add_documents(documents, batch_size=batch_size)
        self.splade_search.add_documents(documents, batch_size=batch_size)
    
    def search(self, query: str, top_k=10, rerank_factor=3):
        """
        Perform hybrid search.
        
        Args:
            query: Query text
            top_k: Number of final results to return
            rerank_factor: Multiple to fetch from individual searches for reranking
                          (e.g. if top_k=10 and rerank_factor=3, fetches 30 from each)
        
        Returns:
            List of dictionaries with document and score
        """
        # Get more results than needed from each system for reranking
        fetch_k = min(top_k * rerank_factor, len(self.documents))
        
        # Get results from both systems
        dense_results = self.dense_search.search(query, top_k=fetch_k)
        splade_results = self.splade_search.search(query, top_k=fetch_k)
        
        # Create dictionaries mapping document ID to score
        dense_scores = {result["id"]: result["score"] for result in dense_results}
        splade_scores = {result["id"]: result["score"] for result in splade_results}
        
        # Get union of all document IDs
        all_ids = set(dense_scores.keys()) | set(splade_scores.keys())
        
        # Combine scores
        combined_scores = []
        for doc_id in all_ids:
            # Get dense score (default to 0 if not in results)
            dense_score = dense_scores.get(doc_id, 0.0)
            
            # Get SPLADE score (default to 0 if not in results)
            splade_score = splade_scores.get(doc_id, 0.0)
            
            # Normalize scores if needed
            # (In practice, you might want a more sophisticated normalization)
            
            # Calculate hybrid score
            hybrid_score = self.alpha * splade_score + (1 - self.alpha) * dense_score
            
            combined_scores.append((doc_id, hybrid_score))
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k results
        results = []
        for i, (doc_id, score) in enumerate(combined_scores[:top_k]):
            results.append({
                "document": self.documents[doc_id],
                "score": score,
                "id": doc_id,
                "dense_score": dense_scores.get(doc_id, 0.0),
                "splade_score": splade_scores.get(doc_id, 0.0)
            })
        
        return results
    
    def tune_alpha(self, queries: List[str], relevant_docs: List[List[int]], 
                  alphas=None, metric="ndcg@10"):
        """
        Tune the alpha parameter for optimal performance.
        
        Args:
            queries: List of evaluation queries
            relevant_docs: List of lists, each containing relevant document IDs for each query
            alphas: List of alpha values to try (default: 0.1 to 0.9 by 0.1)
            metric: Evaluation metric ('ndcg@k' or 'precision@k')
            
        Returns:
            Best alpha value
        """
        if alphas is None:
            alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Parse metric
        if "@" in metric:
            metric_name, k = metric.split("@")
            k = int(k)
        else:
            metric_name = metric
            k = 10
        
        best_score = -1
        best_alpha = 0.5
        
        for alpha in alphas:
            self.alpha = alpha
            total_score = 0
            
            for query_idx, query in enumerate(queries):
                # Get results for this query
                results = self.search(query, top_k=k)
                result_ids = [r["id"] for r in results]
                
                # Calculate metric
                if metric_name == "ndcg":
                    score = self._calculate_ndcg(result_ids, relevant_docs[query_idx], k)
                elif metric_name == "precision":
                    score = self._calculate_precision(result_ids, relevant_docs[query_idx], k)
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")
                
                total_score += score
            
            avg_score = total_score / len(queries)
            print(f"Alpha = {alpha}, {metric} = {avg_score:.4f}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_alpha = alpha
        
        print(f"Best alpha: {best_alpha} with {metric} = {best_score:.4f}")
        self.alpha = best_alpha
        return best_alpha
    
    def _calculate_ndcg(self, result_ids, relevant_ids, k):
        """Calculate NDCG@k metric."""
        # Calculate DCG
        dcg = 0
        for i, doc_id in enumerate(result_ids[:k]):
            if doc_id in relevant_ids:
                # Use binary relevance (1 if relevant, 0 if not)
                # Could use graded relevance if available
                dcg += 1 / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        # Calculate ideal DCG
        idcg = 0
        for i in range(min(len(relevant_ids), k)):
            idcg += 1 / np.log2(i + 2)
        
        # Normalize
        ndcg = dcg / idcg if idcg > 0 else 0
        return ndcg
    
    def _calculate_precision(self, result_ids, relevant_ids, k):
        """Calculate Precision@k metric."""
        # Count relevant documents in top-k results
        relevant_count = sum(1 for doc_id in result_ids[:k] if doc_id in relevant_ids)
        
        # Calculate precision
        precision = relevant_count / min(k, len(result_ids))
        return precision
    
    def save(self, filepath: str):
        """
        Save both indexes and metadata.
        
        Args:
            filepath: Base filepath for saving
        """
        # Save dense index
        self.dense_search.save(f"{filepath}_dense")
        
        # Save SPLADE index
        self.splade_search.save(f"{filepath}_splade")
        
        # Save hybrid metadata
        data = {
            "documents": self.documents,
            "alpha": self.alpha
        }
        
        with open(f"{filepath}_hybrid.json", "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str, use_gpu=False):
        """
        Load hybrid search from disk.
        
        Args:
            filepath: Base filepath for loading
            use_gpu: Whether to use GPU
            
        Returns:
            HybridSearch instance
        """
        # Load hybrid metadata
        with open(f"{filepath}_hybrid.json", "r") as f:
            data = json.load(f)
        
        # Load dense and SPLADE indexes
        dense_search = DenseVectorSearch.load(f"{filepath}_dense", use_gpu=use_gpu)
        splade_search = SpladeSearch.load(f"{filepath}_splade", use_gpu=use_gpu)
        
        # Create instance
        instance = cls(
            dense_model_name=dense_search.embedder.model_name,
            splade_model_dir=splade_search.embedder.model_dir,
            use_gpu=use_gpu,
            alpha=data["alpha"]
        )
        
        # Replace components
        instance.dense_search = dense_search
        instance.splade_search = splade_search
        instance.documents = data["documents"]
        
        return instance


# Example usage:
if __name__ == "__main__":
    # Create hybrid search instance
    hybrid_search = HybridSearch(
        dense_model_name="BAAI/bge-small-en-v1.5",
        splade_model_dir="./fine_tuned_splade/splade-model",
        alpha=0.7  # Weight SPLADE more than dense
    )
    
    # Add documents
    documents = [
        "SPLADE is a sparse retrieval model that combines lexical and semantic matching.",
        "Dense retrieval models map queries and documents to continuous vector spaces.",
        "Hybrid search combines multiple retrieval methods for better results.",
        "FAISS is a library for efficient similarity search developed by Facebook AI Research.",
        "Vector databases are specialized systems for storing and querying vector embeddings.",
        "Approximate nearest neighbor search algorithms trade accuracy for speed.",
        "Information retrieval systems help users find relevant information in large collections."
    ]
    
    hybrid_search.add_documents(documents)
    
    # Search
    results = hybrid_search.search("How do retrieval models work?")
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result['document']} (Score: {result['score']:.4f})")
        print(f"   Dense: {result['dense_score']:.4f}, SPLADE: {result['splade_score']:.4f}")
    
    # Tune alpha with example relevance judgments (if available)
    # queries = ["retrieval models", "vector search", "FAISS library"]
    # relevant_docs = [[0, 1, 2], [1, 4, 5], [3, 4, 5]]
    # hybrid_search.tune_alpha(queries, relevant_docs)
```

## Optimizing for Production

For production use, consider these optimizations:

### 1. Memory Efficiency

```python
# For very large collections, use IVF with Product Quantization
def create_large_scale_index(embeddings, d, use_gpu=False):
    # Number of centroids (more = better recall but slower)
    nlist = min(4096, max(4 * int(np.sqrt(embeddings.shape[0])), 256))
    
    # Number of sub-vectors (more = greater compression but less accuracy)
    m = min(64, d // 4)  # Must be a divisor of d
    
    # Create quantizer
    quantizer = faiss.IndexFlatIP(d)
    
    # Create IVFPQ index
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # 8-bit codes
    
    # Train the index
    print("Training index...")
    index.train(embeddings)
    
    # Add vectors
    print("Adding vectors to index...")
    index.add(embeddings)
    
    # Set nprobe
    index.nprobe = min(64, nlist // 4)  # Higher = more accurate but slower
    
    # Move to GPU if requested
    if use_gpu and faiss.get_num_gpus() > 0:
        print("Moving index to GPU...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    return index
```

### 2. Batch Processing for Large Collections

```python
def add_documents_in_batches(documents, embedder, index_file, batch_size=1000):
    """Process and index documents in batches to avoid memory issues."""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{len(documents)//batch_size + 1}")
        
        # Generate embeddings
        batch_embs = embedder.generate_embeddings(batch)
        
        # Convert to numpy array
        embs_array = np.array(batch_embs).astype(np.float32)
        
        # Add to index (or create index on first batch)
        if i == 0:
            # Create and train index on first batch
            index = create_large_scale_index(embs_array, embs_array.shape[1])
            
            # Add first batch
            index.add(embs_array)
        else:
            # Add subsequent batches
            index.add(embs_array)
        
        # Save index periodically
        if (i + batch_size) % (batch_size * 10) == 0:
            print(f"Saving index at {i + batch_size} documents...")
            faiss.write_index(index, f"{index_file}_partial_{i + batch_size}.index")
    
    # Save final index
    faiss.write_index(index, f"{index_file}.index")
    return index
```

### 3. Multilingual Support

```python
def create_multilingual_search(models=None):
    """Create a search system that supports multiple languages."""
    if models is None:
        models = {
            "en": "BAAI/bge-base-en-v1.5",
            "zh": "BAAI/bge-base-zh-v1.5",
            "multilingual": "intfloat/multilingual-e5-large"
        }
    
    # Create search engine for each language
    search_engines = {}
    for lang, model in models.items():
        search_engines[lang] = DenseVectorSearch(model_name=model)
    
    return search_engines
```

### 4. Distributed Processing with Multiple GPUs

For large-scale indexing, you can distribute the work across multiple GPUs:

```python
def distributed_indexing(documents, embedder, ngpus):
    """Distribute indexing across multiple GPUs."""
    # Split documents into chunks for each GPU
    chunks = [documents[i::ngpus] for i in range(ngpus)]
    
    # Create partial indexes on each GPU
    partial_indexes = []
    for i, chunk in enumerate(chunks):
        with faiss.StandardGpuResources() as res:
            # Use GPU i
            embeddings = generate_embeddings_for_chunk(chunk, embedder)
            index = create_index(embeddings, device=i)
            partial_indexes.append(index)
    
    # Merge partial indexes
    index = faiss.IndexFlatIP(embeddings.shape[1])
    for partial in partial_indexes:
        index.add(faiss.extract_index_vectors(partial))
    
    return index
```

## Conclusion

FAISS provides powerful tools for implementing efficient search at scale. In this article, we've shown how to:

1. Implement dense vector search using our TextEmbedder and FAISS
2. Adapt FAISS for SPLADE sparse vectors
3. Create a hybrid search system that combines both approaches

Each approach has its strengths:

- **Dense Vector Search**: Captures semantic meaning and handles paraphrasing
- **SPLADE Search**: Provides better exact matching and handles rare terms
- **Hybrid Search**: Combines the strengths of both approaches

While SPLADE remains our primary focus, these implementations demonstrate how our toolkit can support a variety of search strategies for different use cases and requirements.

By leveraging FAISS, you can scale your search implementation to handle millions or even billions of documents efficiently.

## Further Reading

- [Facebook AI Similarity Search (FAISS) Documentation](https://github.com/facebookresearch/faiss/wiki)
- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)
- [Understanding Approximate Nearest Neighbor Search](https://www.pinecone.io/learn/approximate-nearest-neighbor/)