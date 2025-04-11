#!/usr/bin/env python3
"""
Unified SPLADE Trainer Module

This module provides a comprehensive trainer for SPLADE (SParse Lexical AnD Expansion) models
with all advanced features including:

- Mixed precision training: Accelerates training using lower precision (FP16) where appropriate
- Early stopping: Prevents overfitting by stopping training when validation metrics plateau
- Checkpointing: Saves and loads model state during training for recovery and best model selection
- Resume training: Allows resuming from previous checkpoints for interrupted training runs
- Robust error handling: Handles and recovers from common errors during training
- Detailed logging: Provides comprehensive metrics tracking and progress reporting

It brings together the best components from all previous trainer implementations
into a cohesive, maintainable solution.

The SPLADE (SParse Lexical AnD Expansion) model uses a sparse representation 
that captures lexical matching while also handling term expansion, making it 
powerful for search applications.

Example usage:
    
    # Basic usage
    trainer = UnifiedSpladeTrainer(
        model_name="prithivida/Splade_PP_en_v1",
        output_dir="./fine_tuned_splade_unified",
        train_file="data/training_data.json"
    )
    trainer.train()
    
    # Advanced usage with all features
    trainer = UnifiedSpladeTrainer(
        model_name="prithivida/Splade_PP_en_v1",
        output_dir="./fine_tuned_splade_unified",
        train_file="data/training_data.json",
        val_file="data/validation_data.json",
        learning_rate=3e-5,
        batch_size=16,
        epochs=10,
        use_mixed_precision=True,
        early_stopping_patience=3,
        save_best_only=True
    )
    trainer.train()
"""

import json
import logging
import os
import random
import time
from typing import Dict, Any, Optional, Union, List

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW, get_scheduler

from src.unified.dataset import SpladeDataset
from src.unified.utils import (
    TrainingLogger, catch_and_log_exceptions, validate_dir_exists, EarlyStopping, Checkpointing
)
from src.utils import DataError, ModelError, TrainingError, DeviceError, FileSystemError


class UnifiedSpladeTrainer:
    """
    Unified trainer for SPLADE model fine-tuning with all advanced features.
    
    This class provides a comprehensive solution for training SPLADE models with
    state-of-the-art techniques for improving training efficiency, reliability,
    and model quality.
    
    Features:
    - Mixed precision training for better performance
    - Early stopping to prevent overfitting
    - Checkpointing for saving/resuming training
    - Comprehensive logging
    - Robust error handling
    - Device auto-detection (CPU, CUDA, MPS)
    
    The training process uses contrastive learning with triplets of 
    (query, positive_document, negative_document) to optimize the SPLADE model
    for retrieving relevant documents.
    
    Training metrics tracked include:
    - Total loss: Combined loss used for optimization
    - Rank loss: Contrastive loss for document ranking
    - Query FLOPS loss: Regularization for query vector sparsity
    - Document FLOPS loss: Regularization for document vector sparsity
    """

    def __init__(
            self,
            model_name: str,
            output_dir: str,
            train_file: str,
            val_file: Optional[str] = None,
            learning_rate: float = 5e-5,
            batch_size: int = 8,
            epochs: int = 3,
            lambda_d: float = 0.0001,  # Regularization for document vectors
            lambda_q: float = 0.0001,  # Regularization for query vectors
            max_length: int = 512,
            seed: int = 42,
            device: Optional[str] = None,
            use_mixed_precision: bool = True,  # Parameter to control mixed precision training
            fp16_opt_level: str = "O1",  # Mixed precision optimization level
            log_dir: Optional[str] = None,  # Directory for logs
            
            # Early stopping parameters
            early_stopping_patience: int = 3,
            early_stopping_min_delta: float = 0.0001,
            early_stopping_monitor: str = "val_loss",
            early_stopping_mode: str = "min",
            
            # Checkpointing parameters
            save_best_only: bool = True,
            save_freq: Optional[int] = 1,
            max_checkpoints: Optional[int] = 3,
            
            # Recovery parameters
            resume_from_checkpoint: Optional[str] = None,
            resume_latest: bool = False,
            resume_best: bool = False,
            
            logger: Optional[logging.Logger] = None  # Logger instance
    ):
        """
        Initialize unified SPLADE trainer.

        Args:
            model_name: Name or path of pre-trained model. Can be a Hugging Face model ID
                       (e.g., "prithivida/Splade_PP_en_v1") or a local directory containing
                       a saved model.
                       
            output_dir: Directory to save fine-tuned model. Will be created if it doesn't exist.
                       The final model will be saved in a subdirectory called "final_model".
                       
            train_file: Path to training data file in JSON format. The file should contain a list
                       of examples, each with "query", "positive_document", and "negative_documents" keys.
                       
            val_file: Path to validation data file in the same format as train_file.
                     If provided, enables validation-based early stopping and best model selection.
                     
            learning_rate: Learning rate for the AdamW optimizer.
                          Higher values may lead to faster convergence but potential instability.
                          Lower values may lead to more stable but slower training.
                          
            batch_size: Batch size for training. Larger batch sizes require more memory but
                       can improve training efficiency. Reduce if experiencing memory issues.
                       
            epochs: Number of training epochs (full passes through the training data).
                   
            lambda_d: Regularization coefficient for document vectors. Controls sparsity
                     of document representations. Higher values produce sparser vectors.
                     
            lambda_q: Regularization coefficient for query vectors. Controls sparsity
                     of query representations. Higher values produce sparser vectors.
                     
            max_length: Maximum sequence length for tokenization. Longer sequences will be truncated.
                       Should be set based on your data characteristics and model constraints.
                       
            seed: Random seed for reproducibility. Affects initialization, data shuffling, etc.
            
            device: Device to run on ('cuda', 'cpu', 'mps', or None for auto-detection).
                   If None, will use CUDA if available, then MPS (on macOS), then CPU.
                   
            use_mixed_precision: Whether to use mixed precision training (FP16).
                                Significantly speeds up training on CUDA devices with Tensor Cores.
                                Automatically disabled if not supported by the selected device.
                                
            fp16_opt_level: Optimization level for mixed precision training.
                           "O1" is the recommended default balancing speed and precision.
                           
            log_dir: Directory for logs. If None, defaults to a "logs" subdirectory in output_dir.
            
            # Early stopping parameters
            early_stopping_patience: Number of epochs with no improvement after which training will stop.
                                    Higher values allow more chances for improvement before stopping.
                                    
            early_stopping_min_delta: Minimum change to qualify as improvement.
                                     Must improve by at least this amount to reset patience counter.
                                     
            early_stopping_monitor: Metric to monitor for early stopping. Typically "val_loss"
                                   requires validation data. Could also use "train_loss" without
                                   validation data, but this is less effective at preventing overfitting.
                                   
            early_stopping_mode: 'min' or 'max' for the monitored metric.
                                'min' for loss metrics (lower is better).
                                'max' for accuracy/score metrics (higher is better).
            
            # Checkpointing parameters
            save_best_only: Whether to save only the best model based on the monitored metric.
                           If True, saves disk space but keeps only the best model.
                           If False, saves a checkpoint after every save_freq epochs.
                           
            save_freq: Frequency of checkpoints in epochs. If save_best_only=True,
                      this parameter is only used to determine when to evaluate the model.
                      
            max_checkpoints: Maximum number of checkpoints to keep. Older checkpoints are removed.
                            Prevents excessive disk usage during long training runs.
            
            # Recovery parameters
            resume_from_checkpoint: Path to specific checkpoint to resume from.
                                   Allows continuing training from an exact point.
                                   
            resume_latest: Whether to resume from the latest checkpoint in the output directory.
                          Useful for continuing after unexpected interruptions.
                          
            resume_best: Whether to resume from the best checkpoint based on the monitored metric.
                        Useful for fine-tuning an already trained model.
            
            logger: Logger instance. If None, a new TrainingLogger will be created.
        
        Note:
            - Only one of resume_from_checkpoint, resume_latest, or resume_best can be specified.
            - Mixed precision requires CUDA and will be automatically disabled if not available.
            - Early stopping with val_loss monitor requires a validation file.
        """
        # Set up logging
        self.log_dir = log_dir or os.path.join(output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create training logger
        self.logger = logger or TrainingLogger("unified_splade_trainer", self.log_dir)
        
        # Store parameters
        self.model_name = model_name
        self.output_dir = output_dir
        self.train_file = train_file
        self.val_file = val_file
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.max_length = max_length
        self.seed = seed
        self.use_mixed_precision = use_mixed_precision
        self.fp16_opt_level = fp16_opt_level
        
        # Early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_mode = early_stopping_mode
        
        # Checkpointing parameters
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.max_checkpoints = max_checkpoints
        
        # Recovery parameters
        self.resume_from_checkpoint = resume_from_checkpoint
        self.resume_latest = resume_latest
        self.resume_best = resume_best
        
        # Ensure only one resume option is specified
        resume_options = sum([
            self.resume_from_checkpoint is not None,
            self.resume_latest,
            self.resume_best
        ])
        if resume_options > 1:
            raise ValueError("Only one of resume_from_checkpoint, resume_latest, or resume_best can be specified")
        
        # For resuming - will be updated if we resume from a checkpoint
        self.start_epoch = 0
        
        # Set seeds for reproducibility
        self.logger.info(f"Setting random seed to {seed}")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Create output directory with validation
        try:
            validate_dir_exists(output_dir, create=True)
        except FileSystemError as e:
            raise ModelError(f"Cannot create output directory: {output_dir}", e)

        # Set device with error handling
        self._setup_device(device)
        
        # Initialize mixed precision if appropriate
        self._setup_mixed_precision()
        
        # Create early stopping
        self.logger.info("Initializing early stopping")
        self.early_stopping = EarlyStopping(
            monitor=early_stopping_monitor,
            min_delta=early_stopping_min_delta,
            patience=early_stopping_patience,
            mode=early_stopping_mode,
            verbose=True,
            logger=self.logger
        )
        
        # Create checkpointing
        self.logger.info("Initializing checkpointing")
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        self.checkpointing = Checkpointing(
            output_dir=checkpoints_dir,
            save_best_only=save_best_only,
            monitor=early_stopping_monitor,
            mode=early_stopping_mode,
            verbose=True,
            save_freq=save_freq,
            max_checkpoints=max_checkpoints,
            logger=self.logger
        )
        
        # Load model and tokenizer with error handling
        self._load_model_and_tokenizer()
        
        # Optimizer and scheduler will be created in train() method
        # after we know the dataset size and can calculate steps
        self.optimizer = None
        self.scheduler = None
        
        # Log configuration
        self._log_configuration()

    def _setup_device(self, device: Optional[str]) -> None:
        """
        Set up device with error handling and automatic detection.
        
        This method determines the best available device for training,
        with appropriate warnings and fallbacks if the requested device
        is not available or if mixed precision is not supported.
        
        The detection priority is:
        1. User-specified device (if provided)
        2. CUDA GPU (if available)
        3. MPS GPU on macOS (if available)
        4. CPU (fallback)
        
        Args:
            device: Device string ('cuda', 'cpu', 'mps') or None for auto-detection
            
        Raises:
            DeviceError: If there's an error setting up the device
        """
        try:
            if device is None:
                # Auto-detect device
                if torch.backends.mps.is_available():
                    self.logger.info("Using MPS (Metal Performance Shaders) for GPU acceleration.")
                    self.device = torch.device("mps")
                    # Mixed precision is not fully supported on MPS yet
                    if self.use_mixed_precision:
                        self.logger.warning("Mixed precision is not fully supported on MPS. Disabling mixed precision.")
                        self.use_mixed_precision = False
                elif torch.cuda.is_available():
                    self.logger.info("Using CUDA for GPU acceleration.")
                    if torch.cuda.device_count() > 1:
                        self.logger.info(f"Found {torch.cuda.device_count()} CUDA devices. Using device 0.")
                    self.device = torch.device("cuda")
                else:
                    self.logger.info("Using CPU for inference.")
                    self.device = torch.device("cpu")
                    # Mixed precision requires CUDA
                    if self.use_mixed_precision:
                        self.logger.warning("Mixed precision requires CUDA. Disabling mixed precision.")
                        self.use_mixed_precision = False
            else:
                # Use user-specified device
                self.device = torch.device(device)
                if device == "cpu" and self.use_mixed_precision:
                    self.logger.warning("Mixed precision requires CUDA. Disabling mixed precision.")
                    self.use_mixed_precision = False
                elif device == "mps" and self.use_mixed_precision:
                    self.logger.warning("Mixed precision is not fully supported on MPS. Disabling mixed precision.")
                    self.use_mixed_precision = False
                    
            # Log device details for CUDA
            if self.device.type == "cuda":
                device_id = self.device.index or 0
                device_name = torch.cuda.get_device_name(device_id)
                total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # GB
                self.logger.info(f"CUDA device name: {device_name}")
                self.logger.info(f"CUDA device total memory: {total_memory:.2f} GB")
                
        except Exception as e:
            raise DeviceError(f"Error setting up device: {e}", e)

    def _setup_mixed_precision(self) -> None:
        """
        Set up mixed precision training if appropriate.
        
        Mixed precision training uses a combination of float32 and float16 
        precision to speed up training and reduce memory usage. This method
        sets up the GradScaler needed for mixed precision training when using
        PyTorch's automatic mixed precision (AMP).
        
        The method automatically disables mixed precision if:
        1. The device is not CUDA
        2. There's an error initializing the GradScaler
        
        Mixed precision training can significantly speed up training (sometimes 2-3x)
        on modern NVIDIA GPUs with Tensor Cores, while using less memory.
        """
        if self.use_mixed_precision:
            if self.device.type != "cuda":
                self.logger.warning(f"Mixed precision not supported on {self.device.type}. Disabling.")
                self.use_mixed_precision = False
            else:
                self.logger.info(f"Using mixed precision training with opt level: {self.fp16_opt_level}")
                try:
                    # Create gradient scaler for mixed precision training
                    self.scaler = GradScaler()
                    
                    # Log CUDA capabilities to verify Tensor Core support
                    if self.device.type == "cuda":
                        device_id = self.device.index or 0
                        compute_capability = torch.cuda.get_device_capability(device_id)
                        cc_major, cc_minor = compute_capability
                        
                        # Tensor Cores are available on Volta (7.0), Turing (7.5), and Ampere (8.0+) GPUs
                        has_tensor_cores = (cc_major >= 7)
                        
                        if has_tensor_cores:
                            self.logger.info(f"GPU supports Tensor Cores (compute capability {cc_major}.{cc_minor}). Mixed precision will significantly improve performance.")
                        else:
                            self.logger.warning(f"GPU does not support Tensor Cores (compute capability {cc_major}.{cc_minor}). Mixed precision will have limited benefits.")
                            
                except Exception as e:
                    self.logger.error(f"Error initializing GradScaler: {e}")
                    self.logger.warning("Disabling mixed precision training")
                    self.use_mixed_precision = False
                    self.scaler = None
        else:
            self.logger.info("Using full precision training")
            self.scaler = None

    @catch_and_log_exceptions(max_attempts=3, delay=1.0)
    def _load_model_and_tokenizer(self) -> None:
        """
        Load model and tokenizer with retry mechanism.
        
        This method loads the SPLADE model and tokenizer from either:
        1. A local directory path
        2. A Hugging Face model ID
        
        The method uses a retry mechanism to handle transient errors
        that might occur during model loading, such as network issues
        when downloading from Hugging Face.
        
        After loading, the model is moved to the appropriate device (CPU, CUDA, or MPS).
        
        Raises:
            ModelError: If the model cannot be loaded after the maximum number of attempts
        """
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            
            # Try to load from local path first
            if os.path.isdir(self.model_name):
                self.logger.info(f"Loading model from local directory: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            else:
                # Load from Hugging Face
                self.logger.info(f"Loading model from Hugging Face: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            
            # Log model size
            model_size = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Model size: {model_size:,} parameters")
            
            # Move model to device
            self.model.to(self.device)
            
            # Print model architecture summary if in debug mode
            self.logger.debug(f"Model architecture: {self.model}")
            
        except Exception as e:
            raise ModelError(f"Failed to load model: {self.model_name}", e)

    def _create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        """
        Create optimizer and learning rate scheduler.
        
        Args:
            num_training_steps: Total number of training steps (batches)
                               calculated as epochs * steps_per_epoch
        
        The method:
        1. Creates an AdamW optimizer for model parameters
        2. Creates a linear learning rate scheduler with warmup
        
        The scheduler decreases the learning rate linearly from the initial value
        to 0 over the training process, with a warmup period at the beginning.
        
        Warmup steps are calculated as 10% of the total training steps, which
        helps stabilize training in the early phases.
        """
        # Create optimizer with weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            # Default AdamW hyperparameters:
            # betas=(0.9, 0.999),
            # eps=1e-8,
            # weight_decay=0.01
        )
        
        # Create scheduler with linear warmup and decay
        warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
        
        self.logger.info(f"Creating learning rate scheduler with {warmup_steps} warmup steps "
                        f"out of {num_training_steps} total steps")
        
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.logger.debug(f"Initial learning rate: {self.learning_rate}")

    def _log_configuration(self) -> None:
        """
        Log training configuration parameters.
        
        This method logs all relevant configuration parameters to both:
        1. The logger output (console and log file)
        2. The training metrics dictionary for later analysis
        
        This ensures that all training runs are properly documented and
        reproducible.
        """
        config = {
            # Model configuration
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "train_file": self.train_file,
            "val_file": self.val_file,
            
            # Training hyperparameters
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lambda_d": self.lambda_d,
            "lambda_q": self.lambda_q,
            "max_length": self.max_length,
            "seed": self.seed,
            
            # Hardware configuration
            "device": str(self.device),
            
            # Mixed precision configuration
            "mixed_precision": self.use_mixed_precision,
            "fp16_opt_level": self.fp16_opt_level,
            
            # Early stopping configuration
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "early_stopping_monitor": self.early_stopping_monitor,
            "early_stopping_mode": self.early_stopping_mode,
            
            # Checkpointing configuration
            "save_best_only": self.save_best_only,
            "save_freq": self.save_freq,
            "max_checkpoints": self.max_checkpoints,
            
            # Recovery configuration
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "resume_latest": self.resume_latest,
            "resume_best": self.resume_best
        }
        
        # Log training configuration
        self.logger.start_training(config)

    def splade_pooling(self, logits: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply SPLADE pooling to get sparse representations.
        
        SPLADE pooling creates a sparse representation by:
        1. Applying ReLU to logits
        2. Taking log(1 + x) of the result
        3. Performing max-pooling over the sequence dimension
        
        This creates a sparse vector where each dimension corresponds to a term
        in the vocabulary with a weight that indicates its importance.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
                  Output from the masked language model prediction head
                  
            attention_mask: Attention mask [batch_size, seq_len]
                          Indicates which tokens are padding (0) and which are content (1)

        Returns:
            Sparse representation [batch_size, vocab_size]
            A vector where each element represents the weight of a vocabulary term
        """
        # Apply RELU to logits and then LOG(1 + x)
        # This is the SPLADE transformation
        activated = torch.log(1 + torch.relu(logits))

        # Apply max pooling over sequence dimension
        # This gives us one weight per vocabulary term
        if attention_mask is not None:
            # Mask out padding tokens by setting their values to 0
            # This ensures they don't contribute to the max pooling
            mask = attention_mask.unsqueeze(-1).expand_as(activated)
            activated = activated * mask

        # Max pooling over sequence dimension (token positions)
        # For each vocab term, take the max weight across all tokens
        pooled = torch.max(activated, dim=1)[0]
        return pooled

    def compute_flops_loss(self, sparse_rep: torch.Tensor) -> torch.Tensor:
        """
        Compute FLOPS regularization loss to encourage sparsity.
        
        This regularization term penalizes dense vectors to encourage
        sparse representations, which are more efficient for retrieval.
        
        The FLOPS loss is calculated as the sum of L1 norms of the
        sparse vectors in the batch. The L1 norm (sum of absolute values)
        is a direct measure of the vector density.

        Args:
            sparse_rep: Sparse representation [batch_size, vocab_size]
                      Output from the splade_pooling function

        Returns:
            Regularization loss calculated as the sum of L1 norms
        """
        # Sum of L1 norms of sparse vectors
        # Each L1 norm is calculated as the sum of absolute values in the vector
        # For each example in the batch, compute L1 norm, then sum all norms
        l1_norms = torch.norm(sparse_rep, p=1, dim=1)  # [batch_size]
        total_l1 = torch.sum(l1_norms)  # scalar
        return total_l1

    def forward_pass(self, batch: Dict[str, torch.Tensor], mixed_precision: bool = False) -> Dict[str, Any]:
        """
        Perform forward pass and calculate loss for a batch of examples.
        
        This method:
        1. Processes the query, positive document, and negative document through the model
        2. Applies SPLADE pooling to get sparse representations
        3. Computes similarity scores between query and documents
        4. Calculates rank loss and regularization losses
        5. Combines losses for optimization
        
        Args:
            batch: Dictionary containing the batch data with keys:
                  - query_input_ids: Token IDs for queries [batch_size, seq_len]
                  - query_attention_mask: Attention mask for queries [batch_size, seq_len]
                  - positive_input_ids: Token IDs for positive documents [batch_size, seq_len]
                  - positive_attention_mask: Attention mask for positive documents [batch_size, seq_len]
                  - negative_input_ids: Token IDs for negative documents [batch_size, seq_len]
                  - negative_attention_mask: Attention mask for negative documents [batch_size, seq_len]
                  
            mixed_precision: Whether to use mixed precision for forward pass
                           When True, uses float16 for most operations to speed up computation
            
        Returns:
            Dictionary with:
            - loss: Combined loss for optimization
            - rank_loss: Contrastive loss component (margin-based)
            - query_flops: Regularization loss for query vectors
            - doc_flops: Regularization loss for document vectors
            - query_rep: Query sparse representations
            - positive_rep: Positive document sparse representations
            - negative_rep: Negative document sparse representations
            
        Raises:
            TrainingError: If forward pass fails
        """
        try:
            # Context manager for mixed precision
            with autocast(enabled=mixed_precision):
                # Forward pass for query
                query_outputs = self.model(
                    input_ids=batch["query_input_ids"],
                    attention_mask=batch["query_attention_mask"],
                    return_dict=True
                )

                # Forward pass for positive document
                positive_outputs = self.model(
                    input_ids=batch["positive_input_ids"],
                    attention_mask=batch["positive_attention_mask"],
                    return_dict=True
                )

                # Forward pass for negative document
                negative_outputs = self.model(
                    input_ids=batch["negative_input_ids"],
                    attention_mask=batch["negative_attention_mask"],
                    return_dict=True
                )

                # Apply SPLADE pooling to get sparse representations
                query_rep = self.splade_pooling(query_outputs.logits, batch["query_attention_mask"])
                positive_rep = self.splade_pooling(positive_outputs.logits, batch["positive_attention_mask"])
                negative_rep = self.splade_pooling(negative_outputs.logits, batch["negative_attention_mask"])

                # Compute similarity scores (dot products)
                # Higher score = more similar documents
                positive_scores = torch.sum(query_rep * positive_rep, dim=1)  # [batch_size]
                negative_scores = torch.sum(query_rep * negative_rep, dim=1)  # [batch_size]

                # Compute contrastive loss (margin-based)
                # We want positive_scores to be higher than negative_scores by at least margin
                margin = 0.2  # Hyperparameter: minimum gap between positive and negative scores
                # ReLU ensures loss is 0 when margin is satisfied
                rank_loss = torch.mean(torch.relu(margin - positive_scores + negative_scores))

                # Compute regularization losses to encourage sparsity
                query_flops = self.compute_flops_loss(query_rep)
                doc_flops = self.compute_flops_loss(positive_rep)

                # Total loss = rank_loss + regularization terms
                loss = rank_loss + self.lambda_q * query_flops + self.lambda_d * doc_flops

            return {
                "loss": loss,
                "rank_loss": rank_loss.item(),
                "query_flops": query_flops.item(),
                "doc_flops": doc_flops.item(),
                "query_rep": query_rep,
                "positive_rep": positive_rep,
                "negative_rep": negative_rep
            }
        except Exception as e:
            raise TrainingError("Forward pass failed", e)

    @catch_and_log_exceptions()
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train model for one epoch.
        
        This method:
        1. Sets the model to training mode
        2. Iterates through all batches in the training data
        3. Performs forward and backward passes
        4. Updates model parameters
        5. Tracks and reports metrics
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number (0-based)

        Returns:
            Average loss for the epoch
            
        The method handles errors during batch processing, allowing training
        to continue even if some batches fail.
        """
        # Set model to training mode (enables dropout, batch norm updates, etc.)
        self.model.train()
        
        # Initialize metrics
        total_loss = 0
        rank_loss_total = 0
        query_flops_total = 0
        doc_flops_total = 0
        
        # Create progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        # Iterate through batches
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass and compute loss based on precision mode
                if self.use_mixed_precision:
                    # Mixed precision forward pass
                    outputs = self.forward_pass(batch, mixed_precision=True)
                    loss = outputs["loss"]
                    
                    # Backward pass with gradient scaling to prevent underflow
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Full precision forward pass
                    outputs = self.forward_pass(batch, mixed_precision=False)
                    loss = outputs["loss"]
                    
                    # Standard backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # Step scheduler to update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()

                # Update metrics
                total_loss += loss.item()
                rank_loss_total += outputs["rank_loss"]
                query_flops_total += outputs["query_flops"]
                doc_flops_total += outputs["doc_flops"]
                
                # Update progress bar with current metrics
                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "rank_loss": outputs["rank_loss"],
                    "q_flops": outputs["query_flops"],
                    "d_flops": outputs["doc_flops"]
                })
                
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_idx}: {e}")
                # Skip this batch but continue training
                continue

        # Calculate average metrics
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_rank_loss = rank_loss_total / num_batches
        avg_query_flops = query_flops_total / num_batches
        avg_doc_flops = doc_flops_total / num_batches
        
        # Log detailed metrics
        self.logger.info(f"  Rank Loss: {avg_rank_loss:.4f}")
        self.logger.info(f"  Query FLOPS Loss: {avg_query_flops:.4f}")
        self.logger.info(f"  Doc FLOPS Loss: {avg_doc_flops:.4f}")
        
        return avg_loss

    @catch_and_log_exceptions()
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate model on validation data.
        
        This method:
        1. Sets the model to evaluation mode (disables dropout, etc.)
        2. Computes loss on validation data without updating model parameters
        3. Handles gradient tracking and other PyTorch evaluation best practices
        
        Args:
            val_loader: DataLoader for validation data

        Returns:
            Average loss for validation data
            
        The method uses no_grad to disable gradient calculation during evaluation,
        which saves memory and speeds up computation.
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        total_loss = 0
        total_samples = 0

        # Disable gradient calculation for evaluation
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # Forward pass and compute loss (always in full precision for evaluation)
                    outputs = self.forward_pass(batch, mixed_precision=False)
                    loss = outputs["loss"]

                    # Update metrics with batch size weighting
                    # This properly accounts for the last batch that might be smaller
                    batch_size = batch["query_input_ids"].size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    
                except Exception as e:
                    self.logger.error(f"Error during evaluation: {e}")
                    # Skip this batch but continue evaluation
                    continue

        # Calculate weighted average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        return avg_loss

    def _try_resume_training(self) -> bool:
        """
        Try to resume training from a checkpoint.
        
        This method attempts to resume training from:
        1. A specific checkpoint path
        2. The latest checkpoint
        3. The best checkpoint based on validation metrics
        
        It loads:
        - Model weights and configuration
        - Tokenizer configuration
        - Optimizer state (if available)
        - Scheduler state (if available)
        - Training metadata (epoch number, metrics)
        
        Returns:
            True if successfully resumed, False otherwise
            
        If no resume option is specified or the resume attempt fails,
        the method returns False, allowing training to start from scratch.
        """
        # Skip if no resume options are specified
        if not any([self.resume_from_checkpoint, self.resume_latest, self.resume_best]):
            return False
        
        try:
            # Determine which checkpoint to load
            if self.resume_from_checkpoint:
                self.logger.info(f"Attempting to resume from specified checkpoint: {self.resume_from_checkpoint}")
                checkpoint_path = self.resume_from_checkpoint
                checkpoint = self.checkpointing.load(checkpoint_path, self.device)
            elif self.resume_latest:
                self.logger.info("Attempting to resume from latest checkpoint")
                checkpoint = self.checkpointing.load_latest(self.device)
            elif self.resume_best:
                self.logger.info("Attempting to resume from best checkpoint")
                checkpoint = self.checkpointing.load_best(self.device)
            else:
                return False
            
            # Load model and tokenizer
            self.model = checkpoint['model']
            self.tokenizer = checkpoint['tokenizer']
            
            # Update epoch info
            self.start_epoch = checkpoint['epoch']
            
            # Try to load optimizer and scheduler states if available
            training_state = checkpoint['training_state']
            if 'optimizer_state_dict' in training_state and self.optimizer:
                self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
                self.logger.info("Resumed optimizer state")
            
            if 'scheduler_state_dict' in training_state and self.scheduler:
                self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
                self.logger.info("Resumed scheduler state")
            
            self.logger.info(f"Successfully resumed training from epoch {self.start_epoch}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error resuming from checkpoint: {e}")
            self.logger.info("Starting training from scratch")
            return False

    @catch_and_log_exceptions()
    def train(self) -> None:
        """
        Train the SPLADE model with all enhanced features.
        
        This is the main training method that:
        1. Creates datasets and data loaders
        2. Initializes optimizer and scheduler
        3. Tries to resume from checkpoint if specified
        4. Trains for the specified number of epochs
        5. Evaluates on validation data if available
        6. Handles early stopping and checkpointing
        7. Saves the final model and training metrics
        
        The method provides detailed logging throughout the training process
        and handles errors robustly.
        
        Training follows this sequence:
        1. For each epoch:
           a. Train on all batches
           b. Evaluate on validation set (if available)
           c. Save checkpoint
           d. Check for early stopping
        2. Save the final model (using the best weights if validation was used)
        3. Report training statistics
        
        Raises:
            Various exceptions that may occur during training
        """
        try:
            # Create datasets
            self.logger.info("Creating datasets")
            train_dataset = SpladeDataset(
                self.train_file,
                self.tokenizer,
                max_length=self.max_length,
                logger=self.logger
            )

            val_dataset = None
            if self.val_file:
                try:
                    val_dataset = SpladeDataset(
                        self.val_file,
                        self.tokenizer,
                        max_length=self.max_length,
                        logger=self.logger
                    )
                except Exception as e:
                    self.logger.error(f"Error loading validation data: {e}")
                    self.logger.warning("Training will continue without validation")
                    if self.early_stopping_monitor == "val_loss":
                        self.logger.warning("Early stopping disabled as validation data is not available")

            # Create data loaders
            self.logger.info(f"Creating data loaders with batch size {self.batch_size}")
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,       # Shuffle training data
                num_workers=4,      # Number of subprocesses for data loading
                drop_last=False,    # Keep the last batch even if smaller than batch_size
                pin_memory=True,    # Pin memory for faster data transfer to GPU
            )

            val_loader = None
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,   # No need to shuffle validation data
                    num_workers=4,   # Number of subprocesses for data loading
                    drop_last=False, # Keep the last batch even if smaller than batch_size
                    pin_memory=True, # Pin memory for faster data transfer to GPU
                )

            # Create optimizer and scheduler
            num_training_steps = self.epochs * len(train_loader)
            self._create_optimizer_and_scheduler(num_training_steps)
            
            # Try to resume training
            resumed = self._try_resume_training()
            
            # If not resuming, clear checkpoints directory
            if not resumed and os.path.exists(self.checkpointing.output_dir):
                self.logger.info(f"Clearing checkpoints directory: {self.checkpointing.output_dir}")
                for filename in os.listdir(self.checkpointing.output_dir):
                    file_path = os.path.join(self.checkpointing.output_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        self.logger.error(f"Error deleting {file_path}: {e}")

            # Training loop
            self.logger.info(f"Starting training for {self.epochs} epochs")
            self.logger.info(f"Training examples: {len(train_dataset)}")
            if val_dataset:
                self.logger.info(f"Validation examples: {len(val_dataset)}")
                
            start_time = time.time()
            all_metrics = {}
            
            # Continue from start_epoch if resuming
            for epoch in range(self.start_epoch, self.epochs):
                epoch_start_time = time.time()
                self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")

                # Train epoch
                train_loss = self.train_epoch(train_loader, epoch)
                self.logger.info(f"Training loss: {train_loss:.4f}")
                
                epoch_duration = time.time() - epoch_start_time
                self.logger.info(f"Epoch duration: {epoch_duration:.2f} seconds")

                # Evaluate on validation set
                metrics = {
                    "train_loss": train_loss,
                    "epoch_duration": epoch_duration
                }
                
                if val_loader:
                    val_loss = self.evaluate(val_loader)
                    self.logger.info(f"Validation loss: {val_loss:.4f}")
                    metrics["val_loss"] = val_loss

                # Log epoch metrics
                self.logger.log_epoch(epoch + 1, metrics)
                all_metrics[epoch + 1] = metrics
                
                # Save checkpoint
                checkpoint_path = self.checkpointing.save(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    epoch=epoch + 1,
                    metrics=metrics,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler
                )
                
                if checkpoint_path:
                    self.logger.info(f"Saved checkpoint: {checkpoint_path}")
                
                # Check for early stopping if validation data is available
                if val_loader and self.early_stopping:
                    stop_training = self.early_stopping(epoch + 1, metrics)
                    if stop_training:
                        self.logger.info(f"Early stopping triggered after epoch {epoch + 1}")
                        break

            # Save final model to output_dir
            final_output_dir = os.path.join(self.output_dir, "final_model")
            os.makedirs(final_output_dir, exist_ok=True)
            
            try:
                # Try to load the best model for the final save
                if val_loader:
                    self.logger.info("Loading best model based on validation loss")
                    try:
                        best_checkpoint = self.checkpointing.load_best(self.device)
                        self.model = best_checkpoint['model']
                        self.tokenizer = best_checkpoint['tokenizer']
                        best_epoch = best_checkpoint['epoch']
                        best_metrics = best_checkpoint['metrics']
                        self.logger.info(f"Loaded best model from epoch {best_epoch}")
                    except Exception as e:
                        self.logger.error(f"Error loading best model: {e}")
                        self.logger.info("Using current model as final")
                        
                # Save the model
                self.model.save_pretrained(final_output_dir)
                self.tokenizer.save_pretrained(final_output_dir)
                
                # Save config for inference
                with open(os.path.join(final_output_dir, "splade_config.json"), "w") as f:
                    json.dump({
                        "model_type": "splade",
                        "base_model_name": self.model_name,
                        "max_length": self.max_length,
                        "pooling": "max",
                        "activation": "log1p_relu",
                        "trained_with_mixed_precision": self.use_mixed_precision
                    }, f, indent=2)
                
                self.logger.info(f"Final model saved to {final_output_dir}")
                
            except Exception as e:
                self.logger.error(f"Error saving final model: {e}")
            
            # Report training statistics
            total_duration = time.time() - start_time
            final_metrics = {
                "total_duration_seconds": total_duration,
                "epochs_completed": epoch + 1,
                "final_train_loss": all_metrics[epoch + 1]["train_loss"]
            }
            
            if val_loader and "val_loss" in all_metrics[epoch + 1]:
                final_metrics["final_val_loss"] = all_metrics[epoch + 1]["val_loss"]
            
            # Add best metrics if available
            if val_loader and hasattr(self.early_stopping, 'best_value'):
                final_metrics["best_val_loss"] = self.early_stopping.best_value
                final_metrics["best_epoch"] = self.early_stopping.best_epoch
            
            self.logger.end_training(final_metrics)
            self.logger.save_metrics()
            
            # Format time for human-readable output
            hours = int(total_duration // 3600)
            minutes = int((total_duration % 3600) // 60)
            seconds = total_duration % 60
            time_str = f"{hours}h {minutes}m {seconds:.2f}s" if hours else f"{minutes}m {seconds:.2f}s"
            
            self.logger.info(f"Training completed in {time_str}")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise
