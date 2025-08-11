# Spatial-Omics GFM Examples

This directory contains example scripts demonstrating how to use the Spatial-Omics Graph Foundation Model for various spatial transcriptomics analysis tasks.

## Available Examples

### 1. Basic Usage Example (`basic_usage_example.py`)

A comprehensive introduction to the Spatial-Omics GFM showing:

- Creating synthetic Visium-like spatial transcriptomics data
- Basic data preprocessing and quality control
- Building spatial graphs from coordinates
- Setting up the Graph Foundation Model
- Running cell type classification
- Basic visualization of results

**Run with:**
```bash
cd examples
python basic_usage_example.py
```

**Expected Output:**
- Console output showing each step of the analysis
- Plots saved as `basic_usage_plots.png` showing spatial data visualization
- Summary of model parameters and performance

### 2. Simple Training Example (`simple_training_example.py`)

Demonstrates how to train the foundation model and task-specific heads:

- Preparing training and validation datasets
- Setting up model configurations
- Implementing a basic training loop
- Evaluating model performance
- Saving trained models

**Run with:**
```bash
cd examples
python simple_training_example.py
```

**Expected Output:**
- Training progress with loss and accuracy metrics
- Trained models saved in `trained_models/` directory
- Final validation accuracy report

## Requirements

Before running the examples, make sure you have installed the package and its dependencies:

```bash
# Install the package in development mode
pip install -e .

# Or install required dependencies manually
pip install torch torch-geometric scanpy squidpy anndata pandas numpy matplotlib scikit-learn
```

## Example Data

The examples use synthetic data generated programmatically to ensure they run without requiring external datasets. The synthetic data mimics real Visium spatial transcriptomics data with:

- Spatial coordinates in a grid-like pattern
- Multiple cell types with distinct expression signatures
- Realistic gene expression patterns
- Proper AnnData formatting

## Understanding the Output

### Basic Usage Example

The basic example will output:
1. **Data Summary**: Number of spots, genes, and preprocessing results
2. **Model Information**: Parameter counts and architecture details
3. **Embeddings**: Shape and characteristics of learned representations  
4. **Predictions**: Cell type classification results (random since untrained)
5. **Visualizations**: Spatial plots showing true vs predicted cell types

### Training Example  

The training example will show:
1. **Training Progress**: Loss values and validation accuracy over epochs
2. **Model Performance**: How accuracy improves during training
3. **Final Results**: Validation accuracy on held-out data
4. **Saved Models**: Location of trained model checkpoints

## Next Steps

After running these examples, consider:

1. **Real Data**: Try with your own Visium, Slide-seq, or Xenium datasets
2. **Different Tasks**: Implement interaction prediction or pathway analysis
3. **Advanced Training**: Use the full training infrastructure in `spatial_omics_gfm.training`
4. **Hyperparameter Tuning**: Optimize model architecture and training parameters
5. **Evaluation**: Compare against baseline methods and published benchmarks

## Troubleshooting

### Common Issues

**ImportError**: Make sure the package is installed properly:
```bash
pip install -e .
```

**CUDA Issues**: The examples will automatically use CPU if CUDA is not available.

**Memory Issues**: Reduce the number of spots/genes in the synthetic data generation:
```python
adata = create_synthetic_visium_data(n_spots=200, n_genes=500)  # Smaller dataset
```

**Visualization Issues**: If plots don't display, make sure you have a proper matplotlib backend:
```bash
pip install matplotlib
# For headless systems:
export MPLBACKEND=Agg
```

### Getting Help

If you encounter issues:

1. Check the console output for error messages
2. Ensure all dependencies are installed correctly
3. Try with smaller datasets first
4. Check the main package documentation
5. Open an issue on the GitHub repository

## Advanced Examples

For more advanced usage patterns, see:

- `task_modules_usage.py`: Demonstrates different task modules
- `visualization_usage.py`: Advanced spatial data visualization
- Main package documentation for production-ready training pipelines

## Performance Notes

The examples are designed for demonstration and may not represent optimal performance:

- Models are kept small for quick execution
- Training epochs are limited  
- Synthetic data may not capture all real-world complexities
- GPU acceleration is used when available but not required

For production use, consider larger models, longer training, and real datasets.