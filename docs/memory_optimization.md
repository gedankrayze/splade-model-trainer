# Memory Optimization for SPLADE Training

This document provides guidance on training SPLADE models in memory-constrained environments, such as laptops, machines without dedicated GPUs, or when working with large models or datasets.

## Overview

The SPLADE model training process can be memory intensive, especially when using larger batch sizes or longer sequence lengths. The `train_splade_optimized.py` script provides options to optimize memory usage during training, allowing successful training even on devices with limited RAM or GPU memory.

## Key Memory Optimization Techniques

The optimized trainer implements several memory-saving techniques:

1. **Gradient Accumulation**: Updates model weights after accumulating gradients over multiple batches, effectively allowing larger "virtual" batch sizes with lower memory footprint.

2. **Device Selection**: Explicitly choose which device to use (CUDA, MPS, or CPU), which can be important when a system has a GPU but with insufficient memory.

3. **Memory Tracking**: Optional monitoring of memory usage during training to help identify and prevent out-of-memory errors.

4. **Memory-Efficient Mode**: Aggressive memory optimizations including more frequent garbage collection and parameter offloading.

5. **Reduced Batch Size**: Training with smaller batches to reduce peak memory usage.

## Usage Examples

### Training on CPU (for MacBooks with MPS memory issues)

If you encounter MPS (Metal Performance Shaders) out-of-memory errors on a MacBook with Apple Silicon:

```bash
python train_splade_optimized.py \
  --train-file data/training.json \
  --output-dir fine_tuned_splade/my-model \
  --device cpu \
  --force-device \
  --batch-size 2 \
  --gradient-accumulation-steps 8
```

This forces CPU training with a small batch size of 2, but an effective batch size of 16 (2 × 8) through gradient accumulation.

### Memory-Constrained GPU Training

For training on a GPU with limited memory:

```bash
python train_splade_optimized.py \
  --train-file data/training.json \
  --output-dir fine_tuned_splade/my-model \
  --device cuda \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --mixed-precision \
  --memory-tracking
```

This uses a smaller batch size with gradient accumulation and mixed precision to reduce memory usage, while tracking memory to monitor usage.

### Extreme Memory Optimization

For environments with severely constrained memory:

```bash
python train_splade_optimized.py \
  --train-file data/training.json \
  --output-dir fine_tuned_splade/my-model \
  --device cpu \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --memory-efficient \
  --memory-tracking
```

This uses the most aggressive memory optimization settings, at the cost of slower training.

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

## Performance Considerations

Memory optimizations often come with performance trade-offs:

- Gradient accumulation increases training time
- CPU training is significantly slower than GPU training
- Memory-efficient mode adds overhead from frequent garbage collection

When possible, it's preferable to use a machine with more memory or a more powerful GPU. However, these optimizations make it possible to train models in resource-constrained environments when necessary.
