# Combining SPLADE with OpenAI-Compatible Embeddings Using FAISS

This article explores how to create a powerful hybrid search system by combining our SPLADE models with embeddings from OpenAI-compatible APIs, including both cloud-based models (OpenAI, Azure OpenAI) and locally-run models via Ollama (such as nomic-embed-text).

## Introduction

Search systems can benefit from combining different approaches to retrieval. In this article, we'll show how to:

1. Generate embeddings using OpenAI-compatible APIs
2. Combine them with our SPLADE models
3. Create a unified search system with FAISS
4. Scale the solution for production use

This approach gives you the best of both worlds: the semantic understanding of large language model embeddings and the lexical precision of SPLADE.

## Setting Up Your Environment

First, let's set up the necessary dependencies:

```bash
# Install OpenAI client and FAISS
pip install openai faiss-cpu

# For GPU acceleration
# pip install faiss-gpu

# Install Ollama (optional, for local embedding models)
# See https://ollama.com/download
```

## 1. Working with OpenAI-Compatible Embeddings

Let's start by creating a class that can generate embeddings from various API sources:

```python
import os
import time
import numpy as np
from typing import List, Dict, Any, Union, Optional
from tqdm import tqdm

# OpenAI client
from openai import OpenAI

# Our SPLADE embedder
from src.embedder import SpladeEmbedder, OpenAiTextEmbedder, EmbedderFactory

class ApiEmbeddingGenerator:
    """Generate embeddings using OpenAI-compatible APIs."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 model_name: str = "text-embedding-3-small",
                 embedding_dim: Optional[int] = None,
                 batch_size: int = 32,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: API key (default: uses OPENAI_API_KEY environment variable)
            api_base: API base URL (default: OpenAI's API, or for local Ollama: http://localhost:11434/v1)
            model_name: Model name (e.g., "text-embedding-3-small", "nomic-embed-text")
            embedding_dim: Output dimension of the embeddings (inferred if not provided)
            batch_size: Batch size for API calls
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries (in seconds)
        """
        # Use environment variable if API key not provided
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set embedding dimension based on model or provided value
        self.embedding_dim = embedding_dim or self._get_default_dimension(model_name)
        
        # Initialize OpenAI client with appropriate base URL
        client_kwargs = {"api_key": self.api_key}
        if self.api_base:
            client_kwargs["base_url"] = self.api_base
            
        # Create client
        self.client = OpenAI(**client_kwargs)
        
        # Create OpenAI-compatible embedder from our toolkit
        self.embedder = OpenAiTextEmbedder(
            openai_client=self.client,
            model_name=self.model_name,
            dimensions=self.embedding_dim
        )
        
        print(f"Initialized embedding generator for {model_name}")
        print(f"API base: {self.api_base or 'default OpenAI'}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def _get_default_dimension(self, model_name: str) -> int:
        """Get default embedding dimension based on model name."""
        # OpenAI models
        if "text-embedding-3-small" in model_name:
            return 1536
        elif "text-embedding-3-large" in model_name:
            return 3072
        elif "text-embedding-ada" in model_name:
            return 1536
        # Ollama/Nomic models
        elif "nomic-embed-text" in model_name:
            return 768
        # Default fallback
        else:
            print(f"Warning: Unknown model {model_name}, defaulting to 768 dimensions")
            return 768
    
    def generate_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show a progress bar
            
        Returns:
            NumPy array of embeddings with shape (len(texts), embedding_dim)
        """
        embeddings = []
        
        # Process in batches to respect API limits
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Generating embeddings with {self.model_name}")
        
        for i in iterator:
            batch = texts[i:i + self.batch_size]
            
            # Try to get embeddings with retries
            for attempt in range(self.max_retries):
                try:
                    batch_embeddings = self.embedder.generate_embeddings(batch)
                    embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        print(f"Error generating embeddings (attempt {attempt+1}): {e}")
                        time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        print(f"Failed after {self.max_retries} attempts: {e}")
                        # Return zeros for this batch as fallback
                        zeros = [[0.0] * self.embedding_dim for _ in range(len(batch))]
                        embeddings.extend(zeros)
        
        return np.array(embeddings).astype(np.float32)


# Example usage:
if __name__ == "__main__":
    # Using OpenAI's embeddings
    generator_openai = ApiEmbeddingGenerator(
        model_name="text-embedding-3-small"
    )
    
    # Using Ollama with nomic-embed-text model
    generator_ollama = ApiEmbeddingGenerator(
        api_base="http://localhost:11434/v1",
        api_key="ollama",  # Ollama doesn't need a real key, but the client requires something
        model_name="nomic-embed-text"
    )
    
    # Example texts
    texts = [
        "SPLADE models create sparse representations for information retrieval.",
        "Dense vectors can capture semantic relationships well.",
        "Hybrid search combines multiple retrieval approaches."
    ]
    
    # Generate embeddings
    try:
        embeddings_openai = generator_openai.generate_embeddings(texts)
        print(f"OpenAI embeddings shape: {embeddings_openai.shape}")
    except Exception as e:
        print(f"OpenAI embeddings failed: {e}")
    
    try:
        embeddings_ollama = generator_ollama.generate_embeddings(texts)
        print(f"Ollama embeddings shape: {embeddings_ollama.shape}")
    except Exception as e:
        print(f"Ollama embeddings failed: {e}")
```

## 2. Building an OpenAI-SPLADE Hybrid Search System

Now let's create a hybrid search system that combines our SPLADE model with OpenAI-compatible embeddings:

```python
import numpy as np
import faiss
import os
import json
from typing import List, Dict, Any, Tuple, Optional

class OpenAiSpladeHybridSearch:
    """Hybrid search combining SPLADE with OpenAI-compatible embeddings."""
    
    def __init__(self, 
                 splade_model_dir: str,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 embedding_model: str = "text-embedding-3-small",
                 alpha: float = 0.5,
                 use_gpu: bool = False):
        """
        Initialize hybrid search.
        
        Args:
            splade_model_dir: Directory containing SPLADE model
            api_key: API key for OpenAI-compatible API
            api_base: Base URL for API (use http://localhost:11434/v1 for Ollama)
            embedding_model: Name of embedding model
            alpha: Weight for combining scores (alpha * splade + (1-alpha) * dense)
            use_gpu: Whether to use GPU for FAISS
        """
        # Initialize SPLADE embedder
        self.splade_embedder = SpladeEmbedder(model_dir=splade_model_dir)
        
        # Initialize API embedder
        self.api_embedder = ApiEmbeddingGenerator(
            api_key=api_key,
            api_base=api_base,
            model_name=embedding_model
        )
        
        # Store parameters
        self.alpha = alpha
        self.use_gpu = use_gpu
        self.embedding_model = embedding_model
        self.splade_model_dir = splade_model_dir
        
        # Initialize indexes
        self.dense_index = None
        self.dense_dim = self.api_embedder.embedding_dim
        
        # For SPLADE, we'll need the vocabulary size
        self.vocab_size = len(self.splade_embedder.tokenizer)
        self.splade_index = None
        
        # Storage for documents
        self.documents = []
        self.document_ids = []  # To support external IDs
        self.index_built = False
    
    def add_documents(self, 
                      documents: List[str], 
                      document_ids: Optional[List[Any]] = None,
                      batch_size: int = 32):
        """
        Add documents to both indexes.
        
        Args:
            documents: List of documents to add
            document_ids: Optional list of external IDs for documents
            batch_size: Batch size for processing
        """
        # Store document IDs or generate sequential IDs
        if document_ids is None:
            start_idx = len(self.documents)
            document_ids = list(range(start_idx, start_idx + len(documents)))
        
        # Store documents and IDs
        self.documents.extend(documents)
        self.document_ids.extend(document_ids)
        
        # Generate dense embeddings from API
        print("Generating dense embeddings from API...")
        dense_vectors = self.api_embedder.generate_embeddings(
            documents, show_progress=True
        )
        
        # Generate SPLADE embeddings
        print("Generating SPLADE embeddings...")
        all_sparse_vectors = []
        for i in tqdm(range(0, len(documents), batch_size), desc="SPLADE encoding"):
            batch = documents[i:i+batch_size]
            batch_sparse = self.splade_embedder.generate_sparse_embeddings(batch)
            all_sparse_vectors.extend(batch_sparse)
        
        # Convert sparse to dense vectors for FAISS
        print("Converting SPLADE vectors to dense format for FAISS...")
        splade_dense = np.zeros((len(all_sparse_vectors), self.vocab_size), dtype=np.float32)
        for i, sparse_vec in enumerate(all_sparse_vectors):
            indices = sparse_vec['indices']
            values = sparse_vec['values']
            
            for idx, val in zip(indices, values):
                if idx < self.vocab_size:
                    splade_dense[i, idx] = val
        
        # Create or update dense index
        if self.dense_index is None:
            print("Creating dense index...")
            self.dense_index = faiss.IndexFlatIP(self.dense_dim)
            
            # Use GPU if requested
            if self.use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.dense_index = faiss.index_cpu_to_gpu(res, 0, self.dense_index)
        
        # Add vectors to dense index
        self.dense_index.add(dense_vectors)
        
        # Create or update SPLADE index
        if self.splade_index is None:
            print("Creating SPLADE index...")
            self.splade_index = faiss.IndexFlatIP(self.vocab_size)
            
            # Use GPU if requested
            if self.use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.splade_index = faiss.index_cpu_to_gpu(res, 0, self.splade_index)
        
        # Add vectors to SPLADE index
        self.splade_index.add(splade_dense)
        
        self.index_built = True
        print(f"Added {len(documents)} documents to the index")
    
    def search(self, query: str, top_k: int = 10, rerank_factor: int = 3) -> List[Dict[str, Any]]:
        """
        Search using the hybrid approach.
        
        Args:
            query: Search query
            top_k: Number of results to return
            rerank_factor: Factor to retrieve from each index (for reranking)
            
        Returns:
            List of results with document, score, and component scores
        """
        if not self.index_built:
            raise ValueError("No documents have been indexed yet")
        
        # Determine how many results to retrieve from each index for reranking
        fetch_k = min(top_k * rerank_factor, len(self.documents))
        
        # Generate dense embedding for the query
        dense_query = self.api_embedder.generate_embeddings([query])[0]
        dense_query = dense_query.reshape(1, -1)
        
        # Generate SPLADE embedding for the query
        splade_query = self.splade_embedder.encode_text(query)
        
        # Convert SPLADE query to dense format for FAISS
        splade_query_dense = np.zeros((1, self.vocab_size), dtype=np.float32)
        for idx, val in zip(splade_query['indices'], splade_query['values']):
            if idx < self.vocab_size:
                splade_query_dense[0, idx] = val
        
        # Search with dense index
        dense_distances, dense_indices = self.dense_index.search(dense_query, fetch_k)
        
        # Search with SPLADE index
        splade_distances, splade_indices = self.splade_index.search(splade_query_dense, fetch_k)
        
        # Create dictionaries mapping document index to score
        dense_scores = {idx: score for idx, score in zip(dense_indices[0], dense_distances[0])}
        splade_scores = {idx: score for idx, score in zip(splade_indices[0], splade_distances[0])}
        
        # Get union of all document indices
        all_indices = set(dense_scores.keys()) | set(splade_scores.keys())
        
        # Combine scores with weighting
        combined_scores = []
        for idx in all_indices:
            dense_score = dense_scores.get(idx, 0.0)
            splade_score = splade_scores.get(idx, 0.0)
            
            # Apply weighting
            hybrid_score = self.alpha * splade_score + (1 - self.alpha) * dense_score
            
            combined_scores.append((idx, hybrid_score, dense_score, splade_score))
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to results format
        results = []
        for i, (idx, score, dense_score, splade_score) in enumerate(combined_scores[:top_k]):
            if 0 <= idx < len(self.documents):  # Check for valid index
                results.append({
                    "rank": i + 1,
                    "document": self.documents[idx],
                    "id": self.document_ids[idx],
                    "score": float(score),
                    "dense_score": float(dense_score),
                    "splade_score": float(splade_score)
                })
        
        return results
    
    def save(self, directory: str):
        """
        Save the search indexes and metadata.
        
        Args:
            directory: Directory to save to
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save dense index
        dense_index_path = os.path.join(directory, "dense_index.faiss")
        
        # If using GPU, move back to CPU for saving
        if self.use_gpu and faiss.get_num_gpus() > 0:
            cpu_index = faiss.index_gpu_to_cpu(self.dense_index)
            faiss.write_index(cpu_index, dense_index_path)
        else:
            faiss.write_index(self.dense_index, dense_index_path)
        
        # Save SPLADE index
        splade_index_path = os.path.join(directory, "splade_index.faiss")
        
        # If using GPU, move back to CPU for saving
        if self.use_gpu and faiss.get_num_gpus() > 0:
            cpu_index = faiss.index_gpu_to_cpu(self.splade_index)
            faiss.write_index(cpu_index, splade_index_path)
        else:
            faiss.write_index(self.splade_index, splade_index_path)
        
        # Save metadata
        metadata = {
            "alpha": self.alpha,
            "embedding_model": self.embedding_model,
            "splade_model_dir": self.splade_model_dir,
            "dense_dim": self.dense_dim,
            "vocab_size": self.vocab_size
        }
        
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        
        # Save documents (potentially large, split into chunks if needed)
        documents_path = os.path.join(directory, "documents.json")
        with open(documents_path, "w") as f:
            json.dump({"documents": self.documents, "document_ids": self.document_ids}, f)
        
        print(f"Saved search index and metadata to {directory}")
    
    @classmethod
    def load(cls, 
             directory: str, 
             api_key: Optional[str] = None,
             api_base: Optional[str] = None,
             use_gpu: bool = False):
        """
        Load search indexes and metadata from directory.
        
        Args:
            directory: Directory containing saved indexes
            api_key: API key (optional, overrides saved configuration)
            api_base: API base URL (optional, overrides saved configuration)
            use_gpu: Whether to use GPU for FAISS
            
        Returns:
            Initialized OpenAiSpladeHybridSearch instance
        """
        # Load metadata
        with open(os.path.join(directory, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(
            splade_model_dir=metadata["splade_model_dir"],
            api_key=api_key,
            api_base=api_base,
            embedding_model=metadata["embedding_model"],
            alpha=metadata["alpha"],
            use_gpu=use_gpu
        )
        
        # Load documents
        with open(os.path.join(directory, "documents.json"), "r") as f:
            data = json.load(f)
            instance.documents = data["documents"]
            instance.document_ids = data["document_ids"]
        
        # Load dense index
        dense_index_path = os.path.join(directory, "dense_index.faiss")
        instance.dense_index = faiss.read_index(dense_index_path)
        
        # Load SPLADE index
        splade_index_path = os.path.join(directory, "splade_index.faiss")
        instance.splade_index = faiss.read_index(splade_index_path)
        
        # Move to GPU if requested
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            instance.dense_index = faiss.index_cpu_to_gpu(res, 0, instance.dense_index)
            instance.splade_index = faiss.index_cpu_to_gpu(res, 0, instance.splade_index)
        
        instance.index_built = True
        print(f"Loaded search index with {len(instance.documents)} documents")
        
        return instance


# Example usage:
if __name__ == "__main__":
    # Create hybrid search
    hybrid_search = OpenAiSpladeHybridSearch(
        splade_model_dir="./fine_tuned_splade/splade-model",
        api_base="http://localhost:11434/v1",  # Remove for OpenAI, set for Ollama
        api_key="ollama",  # Use real API key for OpenAI
        embedding_model="nomic-embed-text",  # For Ollama; use text-embedding-3-small for OpenAI
        alpha=0.6  # Weight slightly in favor of SPLADE
    )
    
    # Add documents
    documents = [
        "SPLADE is a sparse lexical and expansion model for information retrieval.",
        "Dense vector embeddings capture semantic similarity in continuous space.",
        "Hybrid search combines multiple retrieval approaches for better results.",
        "FAISS is a library for efficient similarity search developed by Facebook AI.",
        "Ollama allows running large language models locally on your machine.",
        "The nomic-embed-text model provides quality text embeddings for search.",
        "OpenAI's text-embedding models are widely used for semantic search applications."
    ]
    
    # Add custom IDs (optional)
    doc_ids = ["doc_" + str(i) for i in range(len(documents))]
    
    # Index documents
    hybrid_search.add_documents(documents, document_ids=doc_ids)
    
    # Search with some queries
    queries = [
        "How do embedding models work?",
        "What is SPLADE used for?",
        "Local language model deployment"
    ]
    
    for query in queries:
        results = hybrid_search.search(query, top_k=3)
        print(f"\nQuery: {query}")
        for result in results:
            print(f"[{result['rank']}] {result['document']} (ID: {result['id']})")
            print(f"    Score: {result['score']:.4f} (Dense: {result['dense_score']:.4f}, SPLADE: {result['splade_score']:.4f})")
```

## 3. Using Ollama with Different Models

Ollama provides a convenient way to run embedding models locally. Let's see how to use it with different models:

```python
def test_ollama_models():
    """Test different Ollama embedding models."""
    models = [
        "nomic-embed-text",
        "nomic-embed-text-v1.5",
        "all-minilm"
    ]
    
    text = "This is a test sentence for embedding models."
    
    for model in models:
        print(f"\nTesting Ollama model: {model}")
        try:
            # Create embedder
            embedder = ApiEmbeddingGenerator(
                api_base="http://localhost:11434/v1",
                api_key="ollama",
                model_name=model
            )
            
            # Generate embedding
            start = time.time()
            embedding = embedder.generate_embeddings([text])[0]
            duration = time.time() - start
            
            print(f"Embedding dimension: {len(embedding)}")
            print(f"Generation time: {duration:.2f} seconds")
            print(f"First 5 values: {embedding[:5]}")
        except Exception as e:
            print(f"Error with model {model}: {e}")
```

### Ollama Setup Guide

To use Ollama for local embedding generation:

1. Install Ollama:
   - Mac/Linux: `curl -fsSL https://ollama.com/install.sh | sh`
   - Windows: Download from https://ollama.com/download

2. Pull embedding models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull all-minilm
   ```

3. Test the API:
   ```bash
   curl -X POST http://localhost:11434/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{
       "model": "nomic-embed-text",
       "input": "Hello world"
     }'
   ```

## 4. Scaling for Production Use

For production-scale deployment, we can enhance our implementation with these optimizations:

```python
class ProductionOpenAiSpladeSearch(OpenAiSpladeHybridSearch):
    """Production-ready version with optimizations for scale."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with additional production settings."""
        # IVF parameters for large-scale indexing
        self.nlist = kwargs.pop('nlist', 1024)  # Number of clusters
        self.nprobe = kwargs.pop('nprobe', 64)  # Number of clusters to search
        
        # Quantization parameters for memory efficiency
        self.use_quantization = kwargs.pop('use_quantization', False)
        self.nbits = kwargs.pop('nbits', 8)  # Bits per component
        
        # Call parent init
        super().__init__(*args, **kwargs)
    
    def _create_optimized_index(self, vectors, dimension):
        """Create optimized FAISS index for production scale."""
        # For smaller datasets, use flat index
        if vectors.shape[0] < 10000:
            index = faiss.IndexFlatIP(dimension)
        else:
            # Create IVF index for larger datasets
            quantizer = faiss.IndexFlatIP(dimension)
            
            if self.use_quantization:
                # Use scalar quantization for memory efficiency (IVF with SQ)
                index = faiss.IndexIVFScalarQuantizer(
                    quantizer, dimension, self.nlist,
                    faiss.ScalarQuantizer.QT_8bit,  # 8-bit quantization
                    faiss.METRIC_INNER_PRODUCT
                )
            else:
                # Use IVF without quantization
                index = faiss.IndexIVFFlat(
                    quantizer, dimension, self.nlist,
                    faiss.METRIC_INNER_PRODUCT
                )
            
            # Train the index
            print(f"Training IVF index with {self.nlist} clusters...")
            index.train(vectors)
            
            # Set search parameters
            index.nprobe = self.nprobe
        
        # Add vectors
        index.add(vectors)
        
        # Move to GPU if required
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        return index
    
    def add_documents(self, documents, document_ids=None, batch_size=32):
        """Optimized document addition for production scale."""
        # First collect all embeddings
        print("Generating dense embeddings...")
        dense_vectors = self.api_embedder.generate_embeddings(
            documents, show_progress=True
        )
        
        print("Generating SPLADE embeddings...")
        all_sparse_vectors = []
        for i in tqdm(range(0, len(documents), batch_size), desc="SPLADE encoding"):
            batch = documents[i:i+batch_size]
            batch_sparse = self.splade_embedder.generate_sparse_embeddings(batch)
            all_sparse_vectors.extend(batch_sparse)
        
        # Convert sparse to dense format
        splade_dense = np.zeros((len(all_sparse_vectors), self.vocab_size), dtype=np.float32)
        for i, sparse_vec in enumerate(all_sparse_vectors):
            for idx, val in zip(sparse_vec['indices'], sparse_vec['values']):
                if idx < self.vocab_size:
                    splade_dense[i, idx] = val
        
        # Store document IDs or generate sequential IDs
        if document_ids is None:
            start_idx = len(self.documents)
            document_ids = list(range(start_idx, start_idx + len(documents)))
        
        # Store documents and IDs
        self.documents.extend(documents)
        self.document_ids.extend(document_ids)
        
        # Create optimized indexes
        if self.dense_index is None:
            print("Creating optimized dense index...")
            self.dense_index = self._create_optimized_index(dense_vectors, self.dense_dim)
        else:
            self.dense_index.add(dense_vectors)
        
        if self.splade_index is None:
            print("Creating optimized SPLADE index...")
            self.splade_index = self._create_optimized_index(splade_dense, self.vocab_size)
        else:
            self.splade_index.add(splade_dense)
        
        self.index_built = True
        print(f"Added {len(documents)} documents to production-optimized index")
    
    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Search for multiple queries in batch.
        
        Args:
            queries: List of search queries
            top_k: Number of results to return per query
            
        Returns:
            List of result lists, one per query
        """
        results = []
        
        # Process queries in a batch for API efficiency
        dense_query_batch = self.api_embedder.generate_embeddings(queries)
        
        # Process each query
        for i, query in enumerate(queries):
            # Get dense embedding
            dense_query = dense_query_batch[i].reshape(1, -1)
            
            # Generate SPLADE embedding
            splade_query = self.splade_embedder.encode_text(query)
            
            # Convert SPLADE query to dense
            splade_query_dense = np.zeros((1, self.vocab_size), dtype=np.float32)
            for idx, val in zip(splade_query['indices'], splade_query['values']):
                if idx < self.vocab_size:
                    splade_query_dense[0, idx] = val
            
            # Search with both indexes
            dense_distances, dense_indices = self.dense_index.search(dense_query, top_k)
            splade_distances, splade_indices = self.splade_index.search(splade_query_dense, top_k)
            
            # Combine results (simplified for batch processing - only uses top_k from each)
            dense_results = {idx: score for idx, score in zip(dense_indices[0], dense_distances[0])}
            splade_results = {idx: score for idx, score in zip(splade_indices[0], splade_distances[0])}
            
            # Combine and sort
            all_indices = set(dense_results.keys()) | set(splade_results.keys())
            combined = []
            
            for idx in all_indices:
                dense_score = dense_results.get(idx, 0.0)
                splade_score = splade_results.get(idx, 0.0)
                combined_score = self.alpha * splade_score + (1 - self.alpha) * dense_score
                combined.append((idx, combined_score, dense_score, splade_score))
            
            combined.sort(key=lambda x: x[1], reverse=True)
            
            # Format results
            query_results = []
            for rank, (idx, score, dense_score, splade_score) in enumerate(combined[:top_k]):
                if 0 <= idx < len(self.documents):
                    query_results.append({
                        "rank": rank + 1,
                        "document": self.documents[idx],
                        "id": self.document_ids[idx],
                        "score": float(score),
                        "dense_score": float(dense_score),
                        "splade_score": float(splade_score)
                    })
            
            results.append(query_results)
        
        return results
```

## 5. Integration with Web Services

Here's an example of how to integrate our hybrid search with a Flask web application:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize search system (load from saved file or create new)
search_system = OpenAiSpladeHybridSearch.load("./saved_index")

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    
    query = data.get("query", "")
    top_k = data.get("top_k", 10)
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Run search
    results = search_system.search(query, top_k=top_k)
    
    return jsonify({
        "query": query,
        "results": results
    })

@app.route("/batch_search", methods=["POST"])
def batch_search():
    data = request.json
    
    queries = data.get("queries", [])
    top_k = data.get("top_k", 10)
    
    if not queries:
        return jsonify({"error": "Queries array is required"}), 400
    
    # Run batch search
    results = search_system.search_batch(queries, top_k=top_k)
    
    return jsonify({
        "results": [
            {"query": query, "results": res}
            for query, res in zip(queries, results)
        ]
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
```

## 6. Comparing Different Embedding Models

Let's see how different embedding models affect our hybrid search. We'll compare a few popular options:

```python
import matplotlib.pyplot as plt
import pandas as pd

def compare_embedding_models(test_queries, ground_truth, documents, splade_model_dir):
    """
    Compare different embedding models in hybrid search.
    
    Args:
        test_queries: List of test queries
        ground_truth: List of lists containing ground truth document indices for each query
        documents: List of documents
        splade_model_dir: Directory containing SPLADE model
    """
    # Models to compare
    models = [
        {
            "name": "OpenAI text-embedding-3-small",
            "api_base": None,  # Use default OpenAI URL
            "model": "text-embedding-3-small"
        },
        {
            "name": "OpenAI text-embedding-3-large",
            "api_base": None,  # Use default OpenAI URL
            "model": "text-embedding-3-large"
        },
        {
            "name": "Ollama nomic-embed-text",
            "api_base": "http://localhost:11434/v1",
            "model": "nomic-embed-text"
        },
        {
            "name": "Ollama all-minilm",
            "api_base": "http://localhost:11434/v1",
            "model": "all-minilm"
        }
    ]
    
    # Alpha values to test
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Results
    results = []
    
    for model_info in models:
        print(f"\nTesting {model_info['name']}...")
        
        # Create hybrid search with this model
        search = OpenAiSpladeHybridSearch(
            splade_model_dir=splade_model_dir,
            api_base=model_info["api_base"],
            api_key="ollama" if "ollama" in str(model_info["api_base"]).lower() else None,
            embedding_model=model_info["model"],
            alpha=0.5  # Will be varied in the loop
        )
        
        # Add documents
        search.add_documents(documents)
        
        # Test different alpha values
        for alpha in alphas:
            print(f"  Testing alpha={alpha}...")
            search.alpha = alpha
            
            # Calculate metrics
            ndcg_scores = []
            
            for i, query in enumerate(test_queries):
                search_results = search.search(query, top_k=10)
                result_ids = [res["id"] for res in search_results]
                
                # Calculate NDCG
                ndcg = calculate_ndcg(result_ids, ground_truth[i], k=10)
                ndcg_scores.append(ndcg)
            
            # Average NDCG
            avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
            
            results.append({
                "model": model_info["name"],
                "alpha": alpha,
                "ndcg": avg_ndcg
            })
    
    # Create dataframe for plotting
    df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for model_name in df["model"].unique():
        model_data = df[df["model"] == model_name]
        plt.plot(model_data["alpha"], model_data["ndcg"], marker='o', label=model_name)
    
    plt.xlabel("Alpha (SPLADE weight)")
    plt.ylabel("NDCG@10")
    plt.title("Comparison of Embedding Models in Hybrid Search")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("embedding_model_comparison.png")
    
    # Find best configuration
    best_row = df.loc[df["ndcg"].idxmax()]
    print(f"\nBest configuration:")
    print(f"Model: {best_row['model']}")
    print(f"Alpha: {best_row['alpha']}")
    print(f"NDCG: {best_row['ndcg']:.4f}")
    
    return df

def calculate_ndcg(result_ids, relevant_ids, k=10):
    """Calculate NDCG@k metric."""
    # Calculate DCG
    dcg = 0
    for i, doc_id in enumerate(result_ids[:k]):
        if doc_id in relevant_ids:
            # Binary relevance
            dcg += 1 / np.log2(i + 2)
    
    # Calculate ideal DCG
    idcg = 0
    for i in range(min(len(relevant_ids), k)):
        idcg += 1 / np.log2(i + 2)
    
    # Return NDCG
    return dcg / idcg if idcg > 0 else 0
```

## Conclusion

Combining SPLADE with OpenAI-compatible embeddings provides a powerful hybrid search solution. This approach offers several advantages:

1. **Complementary Strengths**: SPLADE excels at lexical matching and handling rare terms, while dense embeddings from models like OpenAI's text-embedding-3 or Nomic's embed-text excel at semantic understanding.

2. **Flexibility**: You can use cloud-based APIs from OpenAI or local models via Ollama, depending on your requirements for cost, privacy, and performance.

3. **Scalability**: With FAISS and the optimizations shown here, this approach can scale to millions of documents.

4. **Tunability**: The alpha parameter allows you to adjust the balance between sparse and dense retrieval based on your specific use case and content type.

The ideal configuration will depend on your specific domain and requirements. For domain-specific retrieval with specialized terminology, you might favor SPLADE (higher alpha value). For more general semantic search, you might lean toward dense embeddings. In most cases, a balanced hybrid approach delivers the best overall performance.

## Resources

- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [Ollama](https://ollama.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [SPLADE Paper](https://arxiv.org/abs/2107.05720)
- [Introduction to Hybrid Search](introduction_to_splade.md)
- [Working with Embeddings](working_with_embeddings.md)
- [Scaling Search with FAISS](scaling_search_with_faiss.md)