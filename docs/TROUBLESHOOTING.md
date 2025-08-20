# Troubleshooting Guide

## Common Issues

### Installation Issues

#### ModuleNotFoundError: No module named 'numpy'
```bash
# Solution: Install dependencies
pip install numpy pandas matplotlib
```

#### Torch not found
```bash
# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues

#### Out of Memory during training
```python
# Reduce batch size
model.train(batch_size=2)

# Enable gradient checkpointing
model.enable_gradient_checkpointing()

# Use mixed precision
model.enable_mixed_precision()
```

### Data Issues

#### Invalid spatial coordinates
```python
# Check coordinate range
print(f"X range: {coords[:, 0].min()} - {coords[:, 0].max()}")
print(f"Y range: {coords[:, 1].min()} - {coords[:, 1].max()}")

# Normalize coordinates if needed
coords = (coords - coords.min()) / (coords.max() - coords.min())
```

### Performance Issues

#### Slow processing
```python
# Enable caching
from spatial_omics_gfm.performance import enable_caching
enable_caching()

# Use batch processing
processor.process_in_batches(data, batch_size=1000)
```

## Debugging

### Enable debug logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Profile performance
```python
from spatial_omics_gfm.performance import profile_performance

with profile_performance() as profiler:
    # Your code here
    result = model.predict(data)

profiler.print_stats()
```

## Getting Help

1. Check this troubleshooting guide
2. Search existing issues on GitHub
3. Create a new issue with:
   - System information
   - Error message
   - Minimal reproducible example
