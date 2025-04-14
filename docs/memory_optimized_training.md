# Using Memory-Optimized SPLADE Training

This guide explains how to use our memory-optimized approach for training SPLADE models on hardware with limited GPU memory or on CPU.

## Overview

Our project now includes two primary training scripts:

1. `train_splade_unified.py` - Standard trainer for environments with sufficient memory resources
2. `train_splade_optimized.py` - Memory-optimized trainer for constrained environments (e.g., laptops, limited GPU memory)

The memory-optimized trainer allows you to successfully train SPLADE models even on systems where you might encounter out-of-memory errors with the standard trainer.

## When to Use Memory-Optimized Training

Consider using the memory-optimized training approach when:

- You're experiencing out-of-memory errors with the standard trainer
- Training on a laptop or system with limited resources
- Using Apple Silicon (M1/M2/M3/M4) hardware with MPS memory constraints
- You need fine-grained control over memory usage
- You want to explicitly control which device (CPU/GPU/MPS) to use

## Key Memory Optimization Techniques

The optimized trainer implements several memory-saving techniques:

1. **Gradient Accumulation**: Updates model weights after accumulating gradients over multiple batches, effectively allowing larger "virtual" batch sizes with lower memory footprint.

2. **Device Selection**: Explicitly choose which device to use (CUDA, MPS, or CPU), which can be important when a system has a GPU but with insufficient memory.

3. **Memory Tracking**: Optional monitoring of memory usage during training to help identify and prevent out-of-memory errors.

4. **Memory-Efficient Mode**: Aggressive memory optimizations including more frequent garbage collection and parameter offloading.

## Usage Examples

### Training on CPU (for MacBooks with MPS memory issues)

If you encounter MPS (Metal Performance Shaders) out-of-memory errors on a MacBook with Apple Silicon:

```bash
python train_splade_optimized.py \
  --train-file training_data/training_data.json \
  --output-dir fine_tuned_splade/my-model \
  --device cpu \
  --force-device \
  --batch-size 2 \
  --gradient-accumulation-steps 8
```

### Memory-Constrained GPU Training

For training on a GPU with limited memory:

```bash
python train_splade_optimized.py \
  --train-file training_data/training_data.json \
  --output-dir fine_tuned_splade/my-model \
  --device cuda \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --mixed-precision \
  --memory-tracking
```

### Extreme Memory Optimization

For environments with severely constrained memory:

```bash
python train_splade_optimized.py \
  --train-file training_data/training_data.json \
  --output-dir fine_tuned_splade/my-model \
  --device cpu \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --memory-efficient \
  --memory-tracking
```

## Memory Optimization Parameters

| Parameter | Description |
|-----------|-------------|
| `--device` | Explicitly choose device: 'cuda', 'mps', or 'cpu' |
| `--force-device` | Force the specified device even if it might not be optimal |
| `--gradient-accumulation-steps` | Accumulate gradients over N steps before updating weights |
| `--memory-tracking` | Track and log memory usage during training |
| `--memory-efficient` | Enable aggressive memory optimizations |
| `--batch-size` | Reduce to lower memory requirements |
| `--mixed-precision` | Use FP16 precision on CUDA GPUs to reduce memory usage |

## Evaluating Trained Models

After training, you can evaluate your model using the evaluation script:

```bash
python src/evaluate_splade.py \
  --model-dir fine_tuned_splade/my-model/final_model \
  --test-file tests/data/evaluation_data.json \
  --device cpu
```

## Troubleshooting Memory Issues

### MPS (Apple Silicon) Out of Memory

If you see errors like:

```
RuntimeError: MPS backend out of memory (MPS allocated: XX.XX GB, other allocations: XXX.XX MB, max allowed: XX.XX GB)
```

Try the following:

1. Switch to CPU training with `--device cpu --force-device`
2. Reduce batch size and increase gradient accumulation steps
3. Reduce sequence length with `--max-length` if your data allows it

### CUDA Out of Memory

If you encounter CUDA OOM errors:

1. Enable mixed precision with `--mixed-precision`
2. Reduce batch size and use gradient accumulation
3. Use memory tracking to monitor usage

### CPU Memory Limitations

If running out of RAM during CPU training:

1. Enable memory-efficient mode with `--memory-efficient`
2. Use the smallest possible batch size (even 1)
3. Increase gradient accumulation steps to maintain effective batch size

## Additional Documentation

For more details, see:
- `docs/memory_optimization.md` - Detailed guide on memory optimization techniques
- `docs/trainer_comparison.md` - Comparison between unified and optimized trainers
