import logging
import time
import os
from typing import List, Union, Optional, Any, Dict

import numpy as np
import torch
from fastembed import TextEmbedding, SparseTextEmbedding
from transformers import AutoTokenizer, AutoModelForMaskedLM


class SpladeEmbedder:
    """
    Text embedder using custom SPLADE model for sparse embeddings.
    """

    def __init__(self, model_dir: str, max_length: int = 512, device: str = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the SPLADE embedder.

        Args:
            model_dir: Directory containing the trained SPLADE model
            max_length: Maximum sequence length for tokenization
            device: Device to run model on ('cuda', 'cpu', 'mps', or None for auto-detection)
            logger: Optional logger instance
        """
        self.model_dir = model_dir
        self.max_length = max_length
        self.logger = logger or logging.getLogger(__name__)

        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                self.logger.info("Using MPS (Metal Performance Shaders) for GPU acceleration.")
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.logger.info("Using CUDA for GPU acceleration.")
                self.device = torch.device("cuda")
            else:
                self.logger.info("Using CPU for inference.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.logger.info(f"Using device: {self.device}")

        # Load tokenizer and model
        self.logger.info(f"Loading SPLADE model from: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Load SPLADE config if available
        self.config = {}
        config_path = os.path.join(model_dir, "splade_config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                self.config = json.load(f)
                self.logger.info(f"Loaded SPLADE config: {self.config}")

    def encode_text(self, text: str) -> Dict[str, List]:
        """
        Encode text into SPLADE sparse representation.

        Args:
            text: Text to encode

        Returns:
            Dictionary with 'indices' and 'values' for sparse representation
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        # Apply SPLADE pooling (log(1 + ReLU(x)))
        logits = outputs.logits
        activated = torch.log(1 + torch.relu(logits))

        # Max pooling over sequence dimension
        # This accounts for attention mask to ignore padding tokens
        attention_expanded = inputs["attention_mask"].unsqueeze(-1).expand_as(activated)
        masked_activated = activated * attention_expanded
        sparse_rep = torch.max(masked_activated, dim=1)[0]

        # Convert to sparse format (only non-zero elements)
        sparse_rep = sparse_rep.squeeze().cpu()

        # Find non-zero elements
        non_zero_indices = torch.nonzero(sparse_rep).squeeze(1).tolist()
        non_zero_values = sparse_rep[non_zero_indices].tolist()

        # Get token strings for debugging if needed
        # token_strings = [self.tokenizer.decode([idx]) for idx in non_zero_indices]

        return {
            'indices': non_zero_indices,
            'values': non_zero_values
        }

    def generate_sparse_embeddings(self, texts: Union[str, List[str]]) -> List[Dict[str, List]]:
        """
        Generate sparse embeddings for the given texts.

        Args:
            texts: String or list of strings to embed

        Returns:
            List of dictionaries with 'indices' and 'values' for sparse representations
        """
        start_time = time.time()

        if isinstance(texts, str):
            texts = [texts]

        results = []
        for text in texts:
            sparse_embedding = self.encode_text(text)
            results.append(sparse_embedding)

        duration_ms = (time.time() - start_time) * 1000
        if self.logger:
            self.logger.debug(f"Generated {len(texts)} SPLADE sparse embeddings in {duration_ms:.2f}ms")

        return results


class TextEmbedder:
    """
    Text embedder using FastEmbed dense models.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str = ".cache"):
        self.embedding_model = TextEmbedding(
            cache_dir=cache_dir,
            model_name=model_name,
        )

    def generate_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        # Convert the generator to a list and then to a numpy array.
        generated_embeddings = np.array(list(self.embedding_model.embed(texts)))
        # Compute the norm along each row (each embedding vector)
        norms = np.linalg.norm(generated_embeddings, axis=1, keepdims=True)
        # Avoid division by zero by replacing any zero norms with 1
        norms[norms == 0] = 1
        # Normalize each embedding vector
        normalized_embeddings = generated_embeddings / norms
        return normalized_embeddings.tolist()


class HybridEmbedder:
    """
    Embedder that combines dense and sparse embeddings for hybrid search.
    """

    def __init__(
            self,
            dense_model_name: str = "snowflake/snowflake-arctic-embed-l",
            sparse_model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions",
            splade_model_dir: str = None,
            cache_dir: str = ".cache",
            normalize: bool = True,
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the hybrid embedder with both dense and sparse models.

        Args:
            dense_model_name: Name of the dense embedding model
            sparse_model_name: Name of the sparse embedding model for fastembed
            splade_model_dir: Directory of custom SPLADE model (if using custom SPLADE)
            cache_dir: Directory to cache the models
            normalize: Whether to normalize the dense embeddings
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.normalize = normalize
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        self.splade_model_dir = splade_model_dir
        self.use_custom_splade = splade_model_dir is not None

        # Initialize dense embedding model
        self.logger.info(f"Initializing dense embedding model: {dense_model_name}")
        self.dense_model = TextEmbedding(
            model_name=dense_model_name,
            cache_dir=cache_dir
        )

        # Initialize either fastembed sparse model or custom SPLADE model
        if self.use_custom_splade:
            self.logger.info(f"Initializing custom SPLADE model from: {splade_model_dir}")
            self.sparse_model = SpladeEmbedder(
                model_dir=splade_model_dir,
                logger=self.logger
            )
        else:
            self.logger.info(f"Initializing fastembed sparse embedding model: {sparse_model_name}")
            self.sparse_model = SparseTextEmbedding(
                model_name=sparse_model_name,
                cache_dir=cache_dir
            )

        self.logger.info("Both models initialized successfully")

    def generate_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate only dense embeddings for compatibility with existing code.

        Args:
            texts: String or list of strings to embed

        Returns:
            List of dense embedding vectors
        """
        start_time = time.time()
        if isinstance(texts, str):
            texts = [texts]

        # Generate dense embeddings
        dense_embeddings = np.array(list(self.dense_model.embed(texts)))

        if self.normalize:
            # Normalize dense embeddings
            norms = np.linalg.norm(dense_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            dense_embeddings = dense_embeddings / norms

        duration_ms = (time.time() - start_time) * 1000
        if self.logger:
            self.logger.debug(f"Generated {len(texts)} dense embeddings in {duration_ms:.2f}ms")

        return dense_embeddings.tolist()

    def generate_hybrid_embeddings(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Generate both dense and sparse embeddings for hybrid search.

        Args:
            texts: String or list of strings to embed

        Returns:
            List of dictionaries, each containing 'dense' and 'sparse' embeddings.
            The sparse embeddings are in the format {
                'indices': array of token IDs,
                'values': array of corresponding weights
            }
        """
        start_time = time.time()
        if isinstance(texts, str):
            texts = [texts]

        # Generate dense embeddings
        dense_embeddings = np.array(list(self.dense_model.embed(texts)))

        if self.normalize:
            # Normalize dense embeddings
            norms = np.linalg.norm(dense_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            dense_embeddings = dense_embeddings / norms

        # Generate sparse embeddings (different handling based on model type)
        if self.use_custom_splade:
            sparse_embeddings = self.sparse_model.generate_sparse_embeddings(texts)
        else:
            # Using fastembed SparseTextEmbedding
            sparse_embeddings_raw = list(self.sparse_model.embed(texts))
            sparse_embeddings = [
                {
                    'indices': sparse.indices.tolist(),
                    'values': sparse.values.tolist()
                }
                for sparse in sparse_embeddings_raw
            ]

        # Combine into hybrid embeddings
        hybrid_embeddings = []
        for i, (dense, sparse) in enumerate(zip(dense_embeddings, sparse_embeddings)):
            hybrid_embeddings.append({
                'dense': dense.tolist(),
                'sparse': sparse
            })

        duration_ms = (time.time() - start_time) * 1000
        if self.logger:
            self.logger.debug(f"Generated {len(texts)} hybrid embeddings in {duration_ms:.2f}ms")

        return hybrid_embeddings

    def generate_sparse_embeddings(self, texts: Union[str, List[str]]) -> List:
        """
        Generate only sparse embeddings.

        Args:
            texts: String or list of strings to embed

        Returns:
            List of sparse embeddings (format depends on model type)
        """
        start_time = time.time()
        if isinstance(texts, str):
            texts = [texts]

        # Generate sparse embeddings with the appropriate model
        if self.use_custom_splade:
            sparse_embeddings = self.sparse_model.generate_sparse_embeddings(texts)
        else:
            # Using fastembed SparseTextEmbedding
            sparse_embeddings = list(self.sparse_model.embed(texts))

        duration_ms = (time.time() - start_time) * 1000
        if self.logger:
            self.logger.debug(f"Generated {len(texts)} sparse embeddings in {duration_ms:.2f}ms")

        return sparse_embeddings

    def convert_sparse_to_dict(self, sparse_embedding) -> Dict[str, List]:
        """
        Convert a sparse embedding to a dictionary format.

        Args:
            sparse_embedding: A sparse embedding, either SparseEmbedding object or dictionary

        Returns:
            Dict with 'indices' and 'values' keys
        """
        # If it's already a dict with the expected format, return it
        if isinstance(sparse_embedding, dict) and 'indices' in sparse_embedding and 'values' in sparse_embedding:
            return sparse_embedding

        # If it's a fastembed SparseEmbedding object
        if hasattr(sparse_embedding, 'indices') and hasattr(sparse_embedding, 'values'):
            return {
                'indices': sparse_embedding.indices.tolist(),
                'values': sparse_embedding.values.tolist()
            }

        # Fallback case (should not happen)
        self.logger.warning("Unknown sparse embedding format encountered")
        return {
            'indices': [],
            'values': []
        }


class OpenAiTextEmbedder:
    """
    Text embedder using OpenAI-compatible embedding APIs.
    Compatible with OpenAI, Groq, Ollama, and other providers following the OpenAI API spec.
    """

    def __init__(
            self,
            openai_client: Any,
            model_name: str = "text-embedding-3-small",
            dimensions: int = 1536,
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the OpenAI-compatible embedder.

        Args:
            openai_client: An OpenAI API compatible client (OpenAI, Groq, Ollama, etc.)
            model_name: Name of the embedding model to use
            dimensions: Dimensions of the embedding vectors
            logger: Optional logger instance
        """
        self.client = openai_client
        self.model_name = model_name
        self.dimensions = dimensions
        self.logger = logger or logging.getLogger(__name__)

    def generate_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for the given texts.

        Args:
            texts: String or list of strings to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        start_time = time.time()

        # Convert single string to list for consistency
        if isinstance(texts, str):
            texts = [texts]

        try:
            # Get embeddings from the OpenAI-compatible client
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )

            # Extract embedding vectors
            embeddings = [data.embedding for data in response.data]

            # Normalize embeddings
            embeddings_array = np.array(embeddings)
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized_embeddings = embeddings_array / norms

            # Log time taken
            duration_ms = (time.time() - start_time) * 1000
            if self.logger:
                self.logger.debug(f"Generated {len(texts)} embeddings in {duration_ms:.2f}ms")

            return normalized_embeddings.tolist()

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating embeddings: {e}")

            # Return zero vectors as fallback
            return [[0.0] * self.dimensions for _ in range(len(texts))]


class EmbedderFactory:
    """
    Factory class to create the appropriate embedder based on configuration.
    """

    @staticmethod
    def create_embedder(
            embedder_type: str = "fast_embed",
            model_name: str = None,
            dense_model_name: str = None,
            sparse_model_name: str = None,
            splade_model_dir: str = None,
            openai_client: Any = None,
            dimensions: int = None,
            cache_dir: str = ".cache",
            logger: Optional[logging.Logger] = None
    ) -> Union[TextEmbedder, OpenAiTextEmbedder, HybridEmbedder, SpladeEmbedder]:
        """
        Create an embedder of the specified type.

        Args:
            embedder_type: 'fast_embed', 'openai', 'hybrid', or 'splade'
            model_name: Name of the embedding model (for fast_embed or openai)
            dense_model_name: Name of the dense model (for hybrid)
            sparse_model_name: Name of the sparse model (for hybrid)
            splade_model_dir: Directory containing SPLADE model (for 'splade' or hybrid)
            openai_client: OpenAI client (required for 'openai' type)
            dimensions: Vector dimensions (for 'openai' type)
            cache_dir: Cache directory
            logger: Optional logger instance

        Returns:
            An embedder instance
        """
        if embedder_type.lower() == "openai":
            if openai_client is None:
                raise ValueError("OpenAI client is required for OpenAI embedder")

            # Default model based on provider
            if model_name is None:
                model_name = "text-embedding-3-small"

            # Default dimensions based on model
            if dimensions is None:
                if "3-small" in model_name:
                    dimensions = 1536
                elif "3-large" in model_name:
                    dimensions = 3072
                elif "ada" in model_name:
                    dimensions = 1536
                else:
                    dimensions = 1536  # Safe default

            return OpenAiTextEmbedder(
                openai_client=openai_client,
                model_name=model_name,
                dimensions=dimensions,
                logger=logger
            )
        elif embedder_type.lower() == "hybrid":
            return HybridEmbedder(
                dense_model_name=dense_model_name or "snowflake/snowflake-arctic-embed-l",
                sparse_model_name=sparse_model_name or "Qdrant/bm42-all-minilm-l6-v2-attentions",
                splade_model_dir=splade_model_dir,  # Will use SPLADE if provided
                cache_dir=cache_dir,
                logger=logger
            )
        elif embedder_type.lower() == "splade":
            if splade_model_dir is None:
                raise ValueError("SPLADE model directory is required for SPLADE embedder")

            return SpladeEmbedder(
                model_dir=splade_model_dir,
                logger=logger
            )
        else:  # Default to fast_embed
            # Default model for fast_embed
            if model_name is None:
                model_name = "BAAI/bge-small-en-v1.5"

            return TextEmbedder(model_name=model_name, cache_dir=cache_dir)