import json
import logging
import os
import time
from typing import List, Union, Optional, Any, Dict

import numpy as np
import torch
import transformers
from fastembed import TextEmbedding, SparseTextEmbedding
from packaging import version
# Assuming SparseEmbedding object structure from fastembed if needed for type hints
# from fastembed.sparse.sparse_embedding import SparseEmbedding # Example import path
from transformers import AutoTokenizer, AutoModelForMaskedLM


# --- Utility Function ---

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Performs L2 normalization on a NumPy array of embeddings.

    Args:
        embeddings: A NumPy array where each row is an embedding vector.

    Returns:
        A NumPy array with L2 normalized embeddings.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero for zero-length vectors
    norms[norms == 0] = 1.0
    return embeddings / norms


# --- Embedder Classes ---

class SpladeEmbedder:
    """
    Text embedder using custom SPLADE model for sparse embeddings.
    """

    def __init__(self, model_dir: str, max_length: int = 512, device: str = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the SPLADE embedder.

        Args:
            model_dir: Directory containing the trained SPLADE model.
            max_length: Maximum sequence length for tokenization.
            device: Device to run model on ('cuda', 'cpu', 'mps', or None for auto-detection).
            logger: Optional logger instance.
        """
        self.model_dir = model_dir
        self.max_length = max_length
        self.logger = logger or logging.getLogger(__name__)

        # Set device
        if device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                # Check if MPS is available and built to avoid runtime errors on some systems
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
        transformers_version = version.parse(transformers.__version__)
        self.logger.info(f"Transformers version: {transformers_version}")

        # Load tokenizer and model
        self.logger.info(f"Loading SPLADE model from: {model_dir}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForMaskedLM.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            self.logger.error(f"Failed to load SPLADE model from {model_dir}: {e}")
            raise  # Re-raise the exception to indicate failure

        # Load SPLADE config if available (currently informational)
        self.config = {}
        config_path = os.path.join(model_dir, "splade_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                    self.logger.info(f"Loaded SPLADE config: {self.config}")
            except Exception as e:
                self.logger.warning(f"Could not load or parse splade_config.json: {e}")

    def encode_text(self, text: str) -> Dict[str, List]:
        """
        Encode a single text into SPLADE sparse representation.

        Args:
            text: Text to encode.

        Returns:
            Dictionary with 'indices' and 'values' for sparse representation.
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
        # Ensure logits are float for operations
        logits = outputs.logits.float()
        activated = torch.log(1 + torch.relu(logits))

        # Max pooling over sequence dimension, considering attention mask
        attention_expanded = inputs["attention_mask"].unsqueeze(-1).expand_as(activated).float()
        masked_activated = activated * attention_expanded
        # Handle potential empty sequences after masking (all zeros)
        # Use max(0) if all values are negative after ReLU, although log(1+relu) >= 0
        sparse_rep = torch.max(masked_activated, dim=1)[0]

        # Convert to sparse format (only non-zero elements)
        sparse_rep = sparse_rep.squeeze().cpu()  # Shape: [vocab_size]

        # Find non-zero elements
        non_zero_indices = torch.nonzero(sparse_rep, as_tuple=False).squeeze(
            1)  # Use as_tuple=False for backward compatibility
        non_zero_values = sparse_rep[non_zero_indices]

        return {
            'indices': non_zero_indices.tolist(),
            'values': non_zero_values.tolist()
        }

    def generate_sparse_embeddings(self, texts: Union[str, List[str]]) -> List[Dict[str, List]]:
        """
        Generate sparse embeddings for the given texts using the custom SPLADE model.

        Args:
            texts: String or list of strings to embed.

        Returns:
            List of dictionaries, each with 'indices' and 'values' for sparse representations.
        """
        start_time = time.time()

        if isinstance(texts, str):
            texts = [texts]

        results = []
        # Note: Consider batching here for significant performance improvement if many texts are provided
        for text in texts:
            try:
                sparse_embedding = self.encode_text(text)
                results.append(sparse_embedding)
            except Exception as e:
                self.logger.error(f"Failed to encode text with SPLADE: '{text[:50]}...'. Error: {e}")
                # Append an empty representation or re-raise, depending on desired behavior
                results.append({'indices': [], 'values': []})

        duration_ms = (time.time() - start_time) * 1000
        if self.logger:
            self.logger.debug(f"Generated {len(texts)} SPLADE sparse embeddings in {duration_ms:.2f}ms")

        return results


class TextEmbedder:
    """
    Text embedder using FastEmbed dense models. Normalizes embeddings by default.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str = ".cache",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the FastEmbed dense embedder.

        Args:
            model_name: Name of the FastEmbed dense model.
            cache_dir: Directory to cache the model.
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = model_name
        self.cache_dir = cache_dir
        try:
            self.embedding_model = TextEmbedding(
                cache_dir=self.cache_dir,
                model_name=self.model_name,
            )
            self.logger.info(f"Initialized FastEmbed dense model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize FastEmbed model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate L2 normalized dense embeddings for the given texts.

        Args:
            texts: String or list of strings to embed.

        Returns:
            List of normalized dense embedding vectors.
        """
        start_time = time.time()
        if isinstance(texts, str):
            texts = [texts]

        try:
            # Generate embeddings (returns a generator)
            generated_embeddings_iter = self.embedding_model.embed(texts)
            # Convert to numpy array
            embeddings_array = np.array(list(generated_embeddings_iter))

            if embeddings_array.size == 0:
                self.logger.warning("Received empty embeddings array from FastEmbed.")
                return [[] for _ in texts]  # Return list of empty lists matching input count

            # Normalize using the utility function
            normalized_embeddings_array = normalize_embeddings(embeddings_array)

            duration_ms = (time.time() - start_time) * 1000
            if self.logger:
                self.logger.debug(f"Generated {len(texts)} dense embeddings in {duration_ms:.2f}ms")

            return normalized_embeddings_array.tolist()

        except Exception as e:
            self.logger.error(f"Error generating dense embeddings with {self.model_name}: {e}")
            # Determine appropriate fallback, maybe empty lists or re-raise
            # For now, return empty lists matching the input structure
            return [[] for _ in texts]


class HybridEmbedder:
    """
    Embedder that combines dense and sparse embeddings for hybrid search.
    Can use FastEmbed sparse models or a custom SpladeEmbedder.
    """

    def __init__(
            self,
            dense_model_name: str = "snowflake/snowflake-arctic-embed-l",
            sparse_model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions",
            splade_model_dir: str = None,
            cache_dir: str = ".cache",
            normalize_dense: bool = True,
            logger: Optional[logging.Logger] = None,
            splade_device: Optional[str] = None  # Allow specifying device for SPLADE
    ):
        """
        Initialize the hybrid embedder with both dense and sparse models.

        Args:
            dense_model_name: Name of the dense embedding model (FastEmbed).
            sparse_model_name: Name of the sparse embedding model (FastEmbed).
                               Used only if splade_model_dir is None.
            splade_model_dir: Directory of custom SPLADE model. If provided, this overrides
                              sparse_model_name and uses SpladeEmbedder.
            cache_dir: Directory to cache the FastEmbed models.
            normalize_dense: Whether to L2 normalize the dense embeddings.
            logger: Optional logger instance.
            splade_device: Optional device override ('cuda', 'cpu', 'mps') for SpladeEmbedder.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.normalize_dense = normalize_dense
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        self.splade_model_dir = splade_model_dir
        self.use_custom_splade = splade_model_dir is not None
        self.cache_dir = cache_dir

        # Initialize dense embedding model (using TextEmbedder internally for consistency)
        self.logger.info(f"Initializing dense embedding model: {dense_model_name}")
        self.dense_model = TextEmbedder(
            model_name=dense_model_name,
            cache_dir=cache_dir,
            logger=self.logger
        )
        # Note: We'll call dense_model.embedding_model.embed directly later if needed,
        # or just use its generate_embeddings (which includes normalization if desired)

        # Initialize either fastembed sparse model or custom SPLADE model
        self.sparse_model: Union[SpladeEmbedder, SparseTextEmbedding]
        if self.use_custom_splade:
            self.logger.info(f"Initializing custom SPLADE model from: {splade_model_dir}")
            self.sparse_model = SpladeEmbedder(
                model_dir=splade_model_dir,
                logger=self.logger,
                device=splade_device  # Pass device preference
            )
        else:
            self.logger.info(f"Initializing fastembed sparse embedding model: {sparse_model_name}")
            try:
                self.sparse_model = SparseTextEmbedding(
                    model_name=sparse_model_name,
                    cache_dir=cache_dir
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize FastEmbed sparse model {sparse_model_name}: {e}")
                raise

        self.logger.info("Hybrid embedder initialized successfully.")

    def _generate_dense_batch(self, texts: List[str]) -> np.ndarray:
        """Internal helper to generate a batch of dense embeddings."""
        try:
            # Use the raw embed method of the underlying fastembed model
            embeddings_iter = self.dense_model.embedding_model.embed(texts)
            embeddings_array = np.array(list(embeddings_iter))

            if embeddings_array.size == 0 and texts:
                self.logger.warning("Received empty dense embeddings array.")
                # Return array of zeros matching expected dimensions
                # Need to know the dimension. Get it from the model config if possible.
                try:
                    # Attempt to get dimensionality (may vary based on fastembed version/model)
                    # This is a potential point of failure if the attribute doesn't exist.
                    output_dim = self.dense_model.embedding_model.model.get_sentence_embedding_dimension()
                except AttributeError:
                    self.logger.warning("Could not determine embedding dimension for fallback.")
                    # Choose a common fallback or raise error
                    output_dim = 768  # Example fallback
                return np.zeros((len(texts), output_dim))
            elif embeddings_array.size == 0 and not texts:
                return np.empty((0, 0))  # Return empty array for empty input

            if self.normalize_dense:
                return normalize_embeddings(embeddings_array)
            else:
                return embeddings_array
        except Exception as e:
            self.logger.error(f"Error generating dense embeddings batch: {e}")
            # Fallback strategy: return zero vectors of an assumed dimension
            # This is risky if the dimension isn't known. Consider raising instead.
            try:
                output_dim = self.dense_model.embedding_model.model.get_sentence_embedding_dimension()
            except AttributeError:
                output_dim = 768  # Example fallback
            return np.zeros((len(texts), output_dim))


    def generate_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate only dense embeddings (normalized if configured).
        Provided for compatibility with interfaces expecting only dense output.

        Args:
            texts: String or list of strings to embed.

        Returns:
            List of dense embedding vectors (normalized if normalize_dense=True).
        """
        start_time = time.time()
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []

        dense_embeddings_array = self._generate_dense_batch(texts)

        duration_ms = (time.time() - start_time) * 1000
        if self.logger:
            self.logger.debug(f"Generated {len(texts)} dense embeddings in {duration_ms:.2f}ms")

        return dense_embeddings_array.tolist()

    def generate_sparse_embeddings(self, texts: Union[str, List[str]]) -> List[Any]:
        """
        Generate only sparse embeddings.

        Args:
            texts: String or list of strings to embed.

        Returns:
            List of sparse embeddings.
            - If using custom SPLADE: List[Dict[str, List]]
            - If using FastEmbed sparse: List[SparseEmbedding] (object from fastembed library)
            Use `convert_sparse_to_dict` to get a consistent dictionary output.
        """
        start_time = time.time()
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []

        sparse_embeddings: List[Any] = []
        try:
            # Generate sparse embeddings with the appropriate model
            if self.use_custom_splade:
                # This already returns List[Dict[str, List]]
                sparse_embeddings = self.sparse_model.generate_sparse_embeddings(texts)
            else:
                # Using fastembed SparseTextEmbedding, returns List[SparseEmbedding objects]
                sparse_embeddings = list(self.sparse_model.embed(texts))

            duration_ms = (time.time() - start_time) * 1000
            if self.logger:
                self.logger.debug(f"Generated {len(texts)} sparse embeddings in {duration_ms:.2f}ms")

        except Exception as e:
            self.logger.error(f"Error generating sparse embeddings: {e}")
            # Fallback: return list of empty representations matching input size
            if self.use_custom_splade:
                sparse_embeddings = [{'indices': [], 'values': []} for _ in texts]
            else:
                # Can't easily create empty SparseEmbedding objects, return empty dicts
                sparse_embeddings = [{'indices': [], 'values': []} for _ in texts]
                self.logger.warning("Returning empty dicts as fallback for FastEmbed sparse error.")

        return sparse_embeddings

    def generate_hybrid_embeddings(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Generate both dense and sparse embeddings for hybrid search.

        Args:
            texts: String or list of strings to embed.

        Returns:
            List of dictionaries, each containing 'dense' (List[float]) and
            'sparse' (Dict[str, List]) embeddings. Sparse embeddings are
            converted to a consistent dictionary format.
        """
        start_time = time.time()
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []

        # --- Generate Dense Embeddings ---
        dense_embeddings_array = self._generate_dense_batch(texts)
        dense_embeddings_list = dense_embeddings_array.tolist()  # Convert early for consistency

        # --- Generate Sparse Embeddings ---
        sparse_embeddings_raw = self.generate_sparse_embeddings(texts)  # Use the dedicated method

        # --- Combine into Hybrid Embeddings ---
        hybrid_embeddings = []
        if len(dense_embeddings_list) != len(sparse_embeddings_raw):
            self.logger.error(
                f"Mismatch between generated dense ({len(dense_embeddings_list)}) and sparse ({len(sparse_embeddings_raw)}) counts. Skipping combination.")
            # Return structure matching expected output, but indicate error.
            # Or could raise an error.
            return [{'dense': [], 'sparse': {'indices': [], 'values': []}} for _ in texts]

        for dense_vec, sparse_repr_raw in zip(dense_embeddings_list, sparse_embeddings_raw):
            # Convert sparse representation to standard dict format if needed
            sparse_repr_dict = self.convert_sparse_to_dict(sparse_repr_raw)
            hybrid_embeddings.append({
                'dense': dense_vec,
                'sparse': sparse_repr_dict
            })

        duration_ms = (time.time() - start_time) * 1000
        if self.logger:
            self.logger.debug(f"Generated {len(texts)} hybrid embeddings in {duration_ms:.2f}ms")

        return hybrid_embeddings

    def convert_sparse_to_dict(self, sparse_embedding: Any) -> Dict[str, List]:
        """
        Convert a potentially varied sparse embedding format to a standard dictionary.

        Handles:
        - Dict format {'indices': ..., 'values': ...} (from SpladeEmbedder or already converted)
        - fastembed.SparseEmbedding object

        Args:
            sparse_embedding: A sparse embedding representation.

        Returns:
            Dict with 'indices' and 'values' keys. Returns empty lists if conversion fails.
        """
        if isinstance(sparse_embedding, dict) and 'indices' in sparse_embedding and 'values' in sparse_embedding:
            # Already in the desired format
            return sparse_embedding

        # Check if it's a fastembed SparseEmbedding object (duck typing)
        if hasattr(sparse_embedding, 'indices') and hasattr(sparse_embedding, 'values') and isinstance(
                sparse_embedding.indices, np.ndarray):
            try:
                return {
                    'indices': sparse_embedding.indices.tolist(),
                    'values': sparse_embedding.values.tolist()
                }
            except Exception as e:
                self.logger.warning(f"Failed to convert SparseEmbedding object to dict: {e}")
                # Fallback to empty dict if conversion fails
                return {'indices': [], 'values': []}

        # Fallback/Unknown format
        self.logger.warning(
            f"Unknown sparse embedding format encountered: {type(sparse_embedding)}. Returning empty dict.")
        return {
            'indices': [],
            'values': []
        }


class OpenAiTextEmbedder:
    """
    Text embedder using OpenAI-compatible embedding APIs.
    Compatible with OpenAI, Groq, Ollama, etc. Normalizes embeddings by default.
    """

    def __init__(
            self,
            openai_client: Any,  # Expecting an initialized client (e.g., OpenAI(), Groq())
            model_name: str = "text-embedding-3-small",
            dimensions: int = 1536,  # Primarily used for fallback zero vectors
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the OpenAI-compatible embedder.

        Args:
            openai_client: An OpenAI API compatible client instance.
            model_name: Name of the embedding model to use (e.g., 'text-embedding-3-small', 'nomic-embed-text').
            dimensions: Expected dimensions of the embedding vectors. Used for creating
                        fallback zero vectors on error.
            logger: Optional logger instance.
        """
        if openai_client is None:
            # Added check here for clarity, although factory also checks
            raise ValueError("OpenAI client instance is required.")
        self.client = openai_client
        self.model_name = model_name
        self.dimensions = dimensions  # Store for potential fallback
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Initialized OpenAI compatible embedder with model: {self.model_name}")

    def generate_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate L2 normalized embeddings for the given texts using the configured API.

        Args:
            texts: String or list of strings to embed.

        Returns:
            List of normalized embedding vectors. Returns lists of zeros on API error.
        """
        start_time = time.time()

        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []

        try:
            # Get embeddings from the OpenAI-compatible client
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
                # Note: Some APIs might support 'dimensions' param here directly
                # e.g., if model_name is 'text-embedding-3-small' for OpenAI
            )

            # Extract embedding vectors
            embeddings = [data.embedding for data in response.data]

            # Convert to numpy array
            embeddings_array = np.array(embeddings).astype(np.float32)  # Ensure float type

            # Normalize embeddings (OpenAI text-embedding-3 are pre-normalized,
            # but applying again is safe and ensures consistency with other models/APIs)
            normalized_embeddings = normalize_embeddings(embeddings_array)

            duration_ms = (time.time() - start_time) * 1000
            if self.logger:
                self.logger.debug(f"Generated {len(texts)} OpenAI embeddings in {duration_ms:.2f}ms")

            return normalized_embeddings.tolist()

        except Exception as e:
            self.logger.error(f"Error generating OpenAI embeddings (model: {self.model_name}): {e}")
            # Fallback: Return zero vectors.
            # Consider if raising an exception might be more appropriate for your application.
            self.logger.warning(f"Returning zero vectors ({self.dimensions} dimensions) as fallback.")
            return [[0.0] * self.dimensions for _ in range(len(texts))]


# --- Factory Class ---

class EmbedderFactory:
    """
    Factory class to create the appropriate embedder based on configuration.
    """

    @staticmethod
    def create_embedder(
            embedder_type: str = "fast_embed",  # Default embedder type
            # --- General args ---
            logger: Optional[logging.Logger] = None,
            cache_dir: str = ".cache",
            # --- FastEmbed / Dense args ---
            model_name: str = None,  # Used by fast_embed (dense)
            # --- Hybrid args ---
            dense_model_name: str = None,  # Defaults below if None
            sparse_model_name: str = None,  # Defaults below if None (FastEmbed sparse)
            normalize_dense: bool = True,
            # --- SPLADE args ---
            splade_model_dir: str = None,  # If provided, used by 'splade' or 'hybrid'
            splade_device: Optional[str] = None,  # Optional device for SPLADE
            # --- OpenAI args ---
            openai_client: Any = None,
            openai_model_name: str = None,  # Defaults below if None
            openai_dimensions: int = None,  # Defaults below if None

    ) -> Union[TextEmbedder, OpenAiTextEmbedder, HybridEmbedder, SpladeEmbedder]:
        """
        Create an embedder of the specified type with sensible defaults.

        Args:
            embedder_type: 'fast_embed' (dense), 'openai', 'hybrid', or 'splade'. Case-insensitive.
            logger: Optional logger instance.
            cache_dir: Cache directory for downloaded models (FastEmbed, Transformers).

            model_name: Model name for 'fast_embed' type. Defaults to 'BAAI/bge-small-en-v1.5'.

            dense_model_name: Dense model for 'hybrid'. Defaults to 'snowflake/snowflake-arctic-embed-l'.
            sparse_model_name: Sparse model for 'hybrid' (if splade_model_dir is NOT set).
                               Defaults to 'Qdrant/bm42-all-minilm-l6-v2-attentions'.
            normalize_dense: Whether to normalize dense embeddings in 'hybrid'. Defaults to True.

            splade_model_dir: Path to custom SPLADE model. Required for 'splade' type.
                              If provided for 'hybrid' type, it overrides `sparse_model_name`.
            splade_device: Optional device override for SPLADE model ('cuda', 'cpu', 'mps').

            openai_client: Initialized OpenAI-compatible client instance. Required for 'openai' type.
            openai_model_name: Model name for 'openai'. Defaults intelligently based on client or
                               to 'text-embedding-3-small'.
            openai_dimensions: Vector dimensions for 'openai'. Defaults based on model name or 1536.

        Returns:
            An initialized embedder instance based on the configuration.

        Raises:
            ValueError: If required arguments for a specific type are missing.
        """
        embedder_type = embedder_type.lower()
        factory_logger = logger or logging.getLogger(EmbedderFactory.__name__)  # Use factory's logger

        factory_logger.info(f"Creating embedder of type: {embedder_type}")

        if embedder_type == "openai":
            if openai_client is None:
                raise ValueError("`openai_client` instance is required for embedder_type 'openai'")

            # Determine model name default
            final_openai_model = openai_model_name
            if final_openai_model is None:
                # Basic check if client looks like OpenAI's official client
                client_base_url = str(getattr(openai_client, 'base_url', ''))
                if "api.openai.com" in client_base_url:
                    final_openai_model = "text-embedding-3-small"
                else:
                    # For other providers (Ollama, Groq, etc.), a default is harder.
                    # User should specify. Fallback to a common one.
                    final_openai_model = "nomic-embed-text"  # Example, user might need to change
                    factory_logger.warning(
                        f"No `openai_model_name` provided for non-OpenAI client, defaulting to '{final_openai_model}'. Please specify if incorrect.")

            # Determine dimensions default
            final_dimensions = openai_dimensions
            if final_dimensions is None:
                if "3-small" in final_openai_model:
                    final_dimensions = 1536
                elif "3-large" in final_openai_model:
                    final_dimensions = 3072
                elif "ada" in final_openai_model:  # older model
                    final_dimensions = 1536
                elif "nomic" in final_openai_model:  # common ollama model
                    final_dimensions = 768
                else:
                    factory_logger.warning(
                        f"Could not determine default dimensions for OpenAI model '{final_openai_model}'. Falling back to 1536. Please specify `openai_dimensions` if incorrect.")
                    final_dimensions = 1536  # General fallback

            factory_logger.info(f"Using OpenAI model: {final_openai_model}, Dimensions: {final_dimensions}")
            return OpenAiTextEmbedder(
                openai_client=openai_client,
                model_name=final_openai_model,
                dimensions=final_dimensions,
                logger=logger
            )

        elif embedder_type == "hybrid":
            final_dense_model = dense_model_name or "snowflake/snowflake-arctic-embed-l"
            final_sparse_model = sparse_model_name or "Qdrant/bm42-all-minilm-l6-v2-attentions"

            factory_logger.info(f"Using Dense model: {final_dense_model}")
            if splade_model_dir:
                factory_logger.info(f"Using Sparse (SPLADE) model from: {splade_model_dir}")
            else:
                factory_logger.info(f"Using Sparse (FastEmbed) model: {final_sparse_model}")

            return HybridEmbedder(
                dense_model_name=final_dense_model,
                sparse_model_name=final_sparse_model,  # Only used if splade_model_dir is None
                splade_model_dir=splade_model_dir,  # Takes precedence if provided
                cache_dir=cache_dir,
                normalize_dense=normalize_dense,
                logger=logger,
                splade_device=splade_device
            )

        elif embedder_type == "splade":
            if splade_model_dir is None:
                raise ValueError("`splade_model_dir` is required for embedder_type 'splade'")

            factory_logger.info(f"Using SPLADE model from: {splade_model_dir}")
            return SpladeEmbedder(
                model_dir=splade_model_dir,
                logger=logger,
                device=splade_device  # Allow device specification
            )

        elif embedder_type == "fast_embed":
            # Default to fast_embed if type is not recognized or explicitly set
            final_model_name = model_name or "BAAI/bge-small-en-v1.5"
            factory_logger.info(f"Using FastEmbed (dense) model: {final_model_name}")
            return TextEmbedder(
                model_name=final_model_name,
                cache_dir=cache_dir,
                logger=logger
            )
        else:
            raise ValueError(
                f"Unknown embedder_type: '{embedder_type}'. Choose from 'fast_embed', 'openai', 'hybrid', 'splade'.")


# --- Example Usage (Optional) ---
if __name__ == "__main__":
    # Basic logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_logger = logging.getLogger("EmbedderTest")

    # --- Test FastEmbed (Dense) ---
    print("\n--- Testing FastEmbed (Dense) ---")
    try:
        dense_embedder = EmbedderFactory.create_embedder(
            embedder_type="fast_embed",
            model_name="BAAI/bge-small-en-v1.5",  # Specify model
            logger=test_logger
        )
        dense_vectors = dense_embedder.generate_embeddings(["This is a test sentence.", "Another test."])
        print(f"Generated {len(dense_vectors)} dense vectors.")
        if dense_vectors:
            print(f"Dimension: {len(dense_vectors[0])}")
            # print(f"First vector (first 5 dims): {dense_vectors[0][:5]}")
    except Exception as e:
        print(f"Error testing FastEmbed: {e}")

    # --- Test SPLADE (Requires a downloaded SPLADE model) ---
    print("\n--- Testing SPLADE (Sparse) ---")
    # IMPORTANT: Replace with the actual path to your downloaded SPLADE model directory
    splade_path = "./models/splade_pp_sd"  # <--- CHANGE THIS PATH
    if os.path.exists(splade_path):
        try:
            splade_embedder = EmbedderFactory.create_embedder(
                embedder_type="splade",
                splade_model_dir=splade_path,
                logger=test_logger
            )
            sparse_vectors = splade_embedder.generate_sparse_embeddings("Search query about sparse models")
            print(f"Generated {len(sparse_vectors)} SPLADE sparse vectors.")
            if sparse_vectors:
                print(
                    f"Example vector: Indices({len(sparse_vectors[0]['indices'])}): {sparse_vectors[0]['indices'][:10]}..., Values: {sparse_vectors[0]['values'][:10]}...")
        except Exception as e:
            print(f"Error testing SPLADE: {e}")
    else:
        print(f"SPLADE model path not found: {splade_path}. Skipping SPLADE test.")

    # --- Test Hybrid (using FastEmbed sparse default) ---
    print("\n--- Testing Hybrid (FastEmbed Sparse Default) ---")
    try:
        hybrid_embedder_fs = EmbedderFactory.create_embedder(
            embedder_type="hybrid",
            dense_model_name="BAAI/bge-small-en-v1.5",  # Use a smaller dense model for faster test
            # sparse_model_name default will be used
            logger=test_logger
        )
        hybrid_vectors = hybrid_embedder_fs.generate_hybrid_embeddings(["Hybrid search combines dense and sparse."])
        print(f"Generated {len(hybrid_vectors)} hybrid vectors (FastEmbed sparse).")
        if hybrid_vectors:
            print(f"Dense dim: {len(hybrid_vectors[0]['dense'])}")
            print(
                f"Sparse example: Indices({len(hybrid_vectors[0]['sparse']['indices'])}): {hybrid_vectors[0]['sparse']['indices'][:10]}..., Values: {hybrid_vectors[0]['sparse']['values'][:10]}...")
            # Test sparse only generation
            sparse_only = hybrid_embedder_fs.generate_sparse_embeddings("Just sparse please")
            print(f"Generated {len(sparse_only)} sparse-only vectors (type: {type(sparse_only[0])})")
            # Convert to dict
            sparse_dict = hybrid_embedder_fs.convert_sparse_to_dict(sparse_only[0])
            print(f"Converted sparse dict: Indices({len(sparse_dict['indices'])}): {sparse_dict['indices'][:10]}...")

    except Exception as e:
        print(f"Error testing Hybrid (FastEmbed sparse): {e}")

    # --- Test Hybrid (using custom SPLADE) ---
    print("\n--- Testing Hybrid (Custom SPLADE) ---")
    if os.path.exists(splade_path):
        try:
            hybrid_embedder_splade = EmbedderFactory.create_embedder(
                embedder_type="hybrid",
                dense_model_name="BAAI/bge-small-en-v1.5",
                splade_model_dir=splade_path,  # Provide SPLADE path
                logger=test_logger
            )
            hybrid_vectors_splade = hybrid_embedder_splade.generate_hybrid_embeddings(
                ["Hybrid using SPLADE this time."])
            print(f"Generated {len(hybrid_vectors_splade)} hybrid vectors (SPLADE sparse).")
            if hybrid_vectors_splade:
                print(f"Dense dim: {len(hybrid_vectors_splade[0]['dense'])}")
                print(
                    f"Sparse example: Indices({len(hybrid_vectors_splade[0]['sparse']['indices'])}): {hybrid_vectors_splade[0]['sparse']['indices'][:10]}..., Values: {hybrid_vectors_splade[0]['sparse']['values'][:10]}...")
        except Exception as e:
            print(f"Error testing Hybrid (SPLADE): {e}")
    else:
        print(f"SPLADE model path not found: {splade_path}. Skipping Hybrid SPLADE test.")

    # --- Test OpenAI (Requires OpenAI client and API key) ---
    # print("\n--- Testing OpenAI ---")
    # try:
    #     from openai import OpenAI # Needs pip install openai
    #     # Make sure OPENAI_API_KEY environment variable is set
    #     client = OpenAI()
    #     openai_embedder = EmbedderFactory.create_embedder(
    #         embedder_type="openai",
    #         openai_client=client,
    #         # openai_model_name="text-embedding-3-small", # Optional, defaults ok
    #         logger=test_logger
    #     )
    #     openai_vectors = openai_embedder.generate_embeddings(["Testing OpenAI API.", "Embeddings via cloud."])
    #     print(f"Generated {len(openai_vectors)} OpenAI vectors.")
    #     if openai_vectors:
    #          print(f"Dimension: {len(openai_vectors[0])}")
    #          # print(f"First vector (first 5 dims): {openai_vectors[0][:5]}")
    # except ImportError:
    #     print("OpenAI library not installed (`pip install openai`). Skipping OpenAI test.")
    # except Exception as e:
    #     print(f"Error testing OpenAI (check API key and client init): {e}")
