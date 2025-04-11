# Unified SPLADE Trainer

The Unified SPLADE Trainer is a comprehensive solution for training SPLADE (SParse Lexical AnD Expansion) models with all advanced features in a single, cohesive interface.

## Features

- **Mixed Precision Training**: Accelerate training with FP16 mixed precision (requires CUDA)
- **Early Stopping**: Prevent overfitting by stopping training when validation metrics stop improving
- **Checkpointing**: Save and resume training progress, with options to keep only the best models
- **Robust Error Handling**: Comprehensive error handling and recovery mechanisms
- **Detailed Logging**: Extensive logging of training metrics and parameters
- **Device Auto-detection**: Automatically select the best available device (CPU, CUDA, MPS)
- **Training Recovery**: Resume training from a specific checkpoint, the latest checkpoint, or the best checkpoint

## Usage

### Basic Usage

```bash
python train_splade_unified.py \
  --train-file data/training_data.json \
  --output-dir ./fine_tuned_splade_unified
```

### Using a Validation Set and Mixed Precision

```bash
python train_splade_unified.py \
  --train-file data/training_data.json \
  --val-file data/validation_data.json \
  --output-dir ./fine_tuned_splade_unified \
  --mixed-precision
```

### With Early Stopping

```bash
python train_splade_unified.py \
  --train-file data/training_data.json \
  --val-file data/validation_data.json \
  --output-dir ./fine_tuned_splade_unified \
  --early-stopping \
  --early-stopping-patience 5 \
  --early-stopping-monitor val_loss \
  --early-stopping-mode min
```

### Saving Checkpoints

```bash
python train_splade_unified.py \
  --train-file data/training_data.json \
  --val-file data/validation_data.json \
  --output-dir ./fine_tuned_splade_unified \
  --save-best-only \
  --max-checkpoints 3
```

### Resume Training

```bash
python train_splade_unified.py \
  --train-file data/training_data.json \
  --val-file data/validation_data.json \
  --output-dir ./fine_tuned_splade_unified \
  --resume-latest
```

## Advanced Configuration

### Training Parameters

- `--model-name`: Pre-trained model name or path (default: "prithivida/Splade_PP_en_v1")
- `--learning-rate`: Learning rate (default: 5e-5)
- `--batch-size`: Batch size (default: 8)
- `--epochs`: Number of training epochs (default: 3)
- `--lambda-d`: Regularization coefficient for document vectors (default: 0.0001)
- `--lambda-q`: Regularization coefficient for query vectors (default: 0.0001)
- `--max-length`: Maximum sequence length (default: 512)
- `--seed`: Random seed for reproducibility (default: 42)

### Device Configuration

- `--device`: Device to run on ('cuda', 'cpu', 'mps', or auto-detect)

### Mixed Precision Configuration

- `--mixed-precision`: Enable mixed precision training
- `--fp16-opt-level`: Mixed precision optimization level (default: "O1")

### Early Stopping Configuration

- `--early-stopping`: Enable early stopping
- `--early-stopping-patience`: Number of epochs with no improvement after which training will stop (default: 3)
- `--early-stopping-min-delta`: Minimum change to qualify as improvement (default: 0.0001)
- `--early-stopping-monitor`: Metric to monitor for early stopping (default: "val_loss")
- `--early-stopping-mode`: Mode for early stopping (default: "min")

### Checkpointing Configuration

- `--save-best-only`: Save only the best model based on validation metric
- `--save-freq`: Save checkpoint every N epochs (default: 1)
- `--max-checkpoints`: Maximum number of checkpoints to keep (default: 3)

### Recovery Configuration

- `--resume-from-checkpoint`: Path to checkpoint to resume from
- `--resume-latest`: Resume from the latest checkpoint
- `--resume-best`: Resume from the best checkpoint

### Logging Configuration

- `--log-dir`: Directory for logs (default: output_dir/logs)
- `--verbose`: Enable verbose logging

## Programmatic Usage

You can also use the trainer programmatically in your own code:

```python
from src.unified import UnifiedSpladeTrainer

# Create trainer
trainer = UnifiedSpladeTrainer(
    model_name="prithivida/Splade_PP_en_v1",
    output_dir="./fine_tuned_splade_unified",
    train_file="data/training_data.json",
    val_file="data/validation_data.json",
    learning_rate=5e-5,
    batch_size=8,
    epochs=3,
    use_mixed_precision=True,
    save_best_only=True
)

# Train model
trainer.train()
```

## Output Structure

After training, the output directory will contain:

- `checkpoints/`: Directory containing checkpoints saved during training
- `final_model/`: The final model (best model if validation data is provided)
- `logs/`: Training logs and metrics

Each checkpoint directory contains:
- The model weights and configuration
- The tokenizer configuration
- A `checkpoint_info.json` file with training metrics and state
