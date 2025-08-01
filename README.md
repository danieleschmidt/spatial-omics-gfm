# Spatial-Omics GFM (Graph Foundation Model)

Train billion-parameter Graph Transformers on spatial transcriptomics data to predict cell-cell signaling pathways. Features comprehensive data loaders for Visium, Slide-seq V2, and 10X Xenium platforms, enabling breakthrough discoveries in tissue organization and cellular communication.

## Overview

Spatial-Omics GFM is the first foundation model specifically designed for spatial transcriptomics data. By treating tissue sections as graphs where cells are nodes and spatial proximity defines edges, we can predict complex cell-cell interactions, discover novel signaling pathways, and understand tissue organization at unprecedented resolution.

## Key Features

- **Billion-Parameter Models**: Scale to massive spatial datasets
- **Multi-Platform Support**: Native loaders for all major spatial platforms
- **Cell-Cell Signaling**: Predict ligand-receptor interactions spatially
- **Tissue Architecture**: Learn hierarchical tissue organization
- **Zero-Shot Transfer**: Apply to new tissue types without retraining
- **Interactive Visualization**: Explore predictions in spatial context

## Installation

```bash
# Basic installation
pip install spatial-omics-gfm

# With GPU support
pip install spatial-omics-gfm[gpu]

# With visualization tools
pip install spatial-omics-gfm[viz]

# Full installation
pip install spatial-omics-gfm[full]

# Development
git clone https://github.com/yourusername/spatial-omics-gfm
cd spatial-omics-gfm
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from spatial_omics_gfm import SpatialGraphTransformer, VisiumDataset

# Load Visium data
dataset = VisiumDataset(
    h5_path='visium_human_brain.h5',
    spatial_path='spatial/',
    normalize=True,
    compute_spatial_neighbors=True,
    n_neighbors=6
)

# Initialize model
model = SpatialGraphTransformer(
    num_genes=dataset.num_genes,
    hidden_dim=1024,
    num_layers=24,
    num_heads=16,
    dropout=0.1,
    pre_trained='spatial-gfm-base'
)

# Fine-tune on specific tissue
model.fine_tune(
    dataset,
    task='cell_type_prediction',
    epochs=10,
    batch_size=4,
    learning_rate=1e-5
)

# Predict cell-cell interactions
interactions = model.predict_interactions(
    dataset,
    interaction_types=['ligand_receptor', 'paracrine', 'juxtacrine'],
    confidence_threshold=0.8
)

# Visualize results
from spatial_omics_gfm.visualization import SpatialPlotter
plotter = SpatialPlotter()
plotter.plot_interactions(
    dataset,
    interactions,
    save_path='cell_interactions.html'
)
```

### Multi-Platform Data Loading

```python
from spatial_omics_gfm.data import (
    VisiumDataset,
    SlideSeqDataset,
    XeniumDataset,
    MERFISHDataset
)

# 10X Visium
visium = VisiumDataset.from_10x_folder(
    'path/to/visium/outs/',
    filter_genes=True,
    min_cells_per_gene=10,
    min_genes_per_cell=200
)

# Slide-seq V2
slideseq = SlideSeqDataset(
    count_matrix='slideseq_counts.csv',
    coordinates='slideseq_coords.csv',
    normalize_method='scran',
    build_spatial_graph=True
)

# 10X Xenium
xenium = XeniumDataset(
    xenium_dir='xenium_output/',
    cell_boundaries=True,
    subcellular_resolution=True,
    merge_fov=True  # Merge fields of view
)

# MERFISH
merfish = MERFISHDataset(
    data_dir='merfish_data/',
    codebook='codebook.csv',
    blank_correction=True,
    z_stack_projection='max'
)
```

## Architecture

```
spatial-omics-gfm/
├── spatial_omics_gfm/
│   ├── models/
│   │   ├── graph_transformer.py      # Core architecture
│   │   ├── spatial_attention.py      # Spatial-aware attention
│   │   ├── hierarchical_pooling.py   # Multi-scale pooling
│   │   └── pretrained_models.py      # Model zoo
│   ├── data/
│   │   ├── loaders/                  # Platform-specific loaders
│   │   │   ├── visium.py
│   │   │   ├── slideseq.py
│   │   │   ├── xenium.py
│   │   │   └── merfish.py
│   │   ├── preprocessing.py          # Data preprocessing
│   │   ├── graph_construction.py     # Spatial graph building
│   │   └── augmentation.py           # Data augmentation
│   ├── tasks/
│   │   ├── cell_typing.py            # Cell type prediction
│   │   ├── interaction_prediction.py  # Cell-cell interactions
│   │   ├── pathway_analysis.py       # Pathway enrichment
│   │   └── tissue_segmentation.py    # Tissue architecture
│   ├── training/
│   │   ├── distributed_training.py   # Multi-GPU training
│   │   ├── curriculum_learning.py    # Progressive training
│   │   ├── contrastive_learning.py   # Self-supervised
│   │   └── fine_tuning.py           # Task adaptation
│   ├── inference/
│   │   ├── batch_inference.py        # Large-scale inference
│   │   ├── streaming_inference.py    # Memory-efficient
│   │   └── uncertainty.py            # Uncertainty estimation
│   └── visualization/
│       ├── spatial_plots.py          # Spatial visualization
│       ├── interaction_networks.py   # Network viz
│       └── pathway_maps.py           # Pathway visualization
├── pretrained_models/
├── examples/
└── benchmarks/
```

## Model Architecture

### Spatial Graph Transformer

```python
from spatial_omics_gfm.models import SpatialGraphTransformer

class SpatialGraphTransformer(nn.Module):
    """
    Billion-parameter transformer for spatial transcriptomics
    """
    def __init__(
        self,
        num_genes: int,
        hidden_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        spatial_encoding_dim: int = 64,
        max_neighbors: int = 10
    ):
        super().__init__()
        
        # Gene expression encoder
        self.gene_encoder = nn.Linear(num_genes, hidden_dim)
        
        # Spatial position encoding
        self.spatial_encoder = SpatialPositionEncoding(
            spatial_dim=2,  # 2D for most platforms
            encoding_dim=spatial_encoding_dim
        )
        
        # Graph transformer layers
        self.layers = nn.ModuleList([
            SpatialTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.1,
                use_edge_features=True
            ) for _ in range(num_layers)
        ])
        
        # Task-specific heads
        self.cell_type_head = CellTypeClassifier(hidden_dim)
        self.interaction_head = InteractionPredictor(hidden_dim)
        self.pathway_head = PathwayAnalyzer(hidden_dim)
```

### Spatial Attention Mechanism

```python
from spatial_omics_gfm.models import SpatialAttention

class SpatialAttention(nn.Module):
    """
    Attention mechanism that considers spatial relationships
    """
    def __init__(self, hidden_dim, num_heads, max_distance=500):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Spatial bias
        self.distance_embedding = nn.Embedding(max_distance, num_heads)
        
    def forward(self, x, edge_index, edge_attr=None):
        # Compute attention with spatial bias
        Q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        
        # Add spatial bias based on physical distance
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        
        if edge_attr is not None:
            # edge_attr contains spatial distances
            spatial_bias = self.distance_embedding(edge_attr)
            attention_scores += spatial_bias
        
        return self.aggregate(attention_scores, V, edge_index)
```

## Pre-trained Models

### Model Zoo

```python
from spatial_omics_gfm import load_pretrained_model

# Available models
models = {
    'spatial-gfm-base': {
        'parameters': '350M',
        'training_data': '10M cells from 50 tissues',
        'tasks': ['cell_typing', 'interaction_prediction']
    },
    'spatial-gfm-large': {
        'parameters': '1.3B',
        'training_data': '100M cells from 200 tissues',
        'tasks': ['all']
    },
    'spatial-gfm-xlarge': {
        'parameters': '3B',
        'training_data': '500M cells from 1000 tissues',
        'tasks': ['all', 'zero_shot_transfer']
    }
}

# Load pre-trained model
model = load_pretrained_model(
    'spatial-gfm-large',
    device='cuda',
    precision='float16'  # For memory efficiency
)

# Zero-shot application to new tissue
predictions = model.zero_shot_predict(
    new_tissue_data,
    task='cell_type_annotation',
    confidence_calibration=True
)
```

### Fine-tuning Strategies

```python
from spatial_omics_gfm.training import FineTuner

# Task-specific fine-tuning
fine_tuner = FineTuner(
    base_model='spatial-gfm-large',
    task='tumor_microenvironment_analysis'
)

# Progressive fine-tuning
fine_tuner.progressive_finetune(
    datasets=[small_dataset, medium_dataset, large_dataset],
    epochs_per_stage=[10, 5, 3],
    learning_rates=[1e-4, 5e-5, 1e-5]
)

# Low-rank adaptation (LoRA) for efficient fine-tuning
fine_tuner.lora_finetune(
    dataset,
    rank=16,
    alpha=32,
    target_modules=['q_proj', 'v_proj']
)
```

## Cell-Cell Interaction Prediction

### Ligand-Receptor Analysis

```python
from spatial_omics_gfm.tasks import LigandReceptorPredictor

# Initialize predictor
lr_predictor = LigandReceptorPredictor(
    model=model,
    database='cellphonedb',  # or 'nichenet', 'connectome'
    species='human'
)

# Predict interactions
interactions = lr_predictor.predict(
    spatial_data=dataset,
    min_expression=0.1,
    distance_threshold=200,  # micrometers
    p_value_threshold=0.05
)

# Analyze signaling neighborhoods
neighborhoods = lr_predictor.find_signaling_neighborhoods(
    interactions,
    min_size=5,
    enrichment_method='spatial_permutation'
)

# Visualize communication networks
lr_predictor.plot_communication_network(
    interactions,
    cell_types=['T_cells', 'Macrophages', 'Tumor_cells'],
    layout='force_directed',
    save_path='communication_network.pdf'
)
```

### Spatial Pathway Analysis

```python
from spatial_omics_gfm.tasks import SpatialPathwayAnalyzer

analyzer = SpatialPathwayAnalyzer(
    model=model,
    pathway_database='kegg'  # or 'reactome', 'go'
)

# Spatially-resolved pathway activity
pathway_activity = analyzer.compute_pathway_activity(
    dataset,
    method='graph_propagation',
    normalize=True
)

# Find spatially co-regulated pathways
coregulated = analyzer.find_coregulated_pathways(
    pathway_activity,
    correlation_threshold=0.7,
    spatial_coherence=True
)

# Pathway gradient analysis
gradients = analyzer.analyze_pathway_gradients(
    pathway_activity,
    reference_point='tumor_center',
    n_rings=10
)
```

## Advanced Features

### Multi-Resolution Analysis

```python
from spatial_omics_gfm.models import MultiResolutionGFM

# Hierarchical model for multi-scale analysis
multi_res_model = MultiResolutionGFM(
    base_model=model,
    resolutions=['cell', 'niche', 'region', 'tissue'],
    pooling_method='learnable'
)

# Analyze at multiple scales
multi_scale_results = multi_res_model.analyze(
    dataset,
    tasks={
        'cell': 'type_prediction',
        'niche': 'microenvironment_classification',
        'region': 'functional_annotation',
        'tissue': 'disease_state'
    }
)

# Cross-scale interactions
cross_scale = multi_res_model.find_cross_scale_patterns(
    multi_scale_results,
    min_correlation=0.6
)
```

### Temporal-Spatial Modeling

```python
from spatial_omics_gfm.models import TemporalSpatialGFM

# For time-series spatial data
ts_model = TemporalSpatialGFM(
    spatial_model=model,
    temporal_encoding='lstm',
    num_timepoints=10
)

# Load temporal data
temporal_dataset = load_temporal_spatial_data(
    timepoints=['0h', '6h', '12h', '24h', '48h'],
    aligned=True
)

# Predict dynamics
dynamics = ts_model.predict_dynamics(
    temporal_dataset,
    interpolate_missing=True,
    smooth_transitions=True
)

# Find temporal patterns
patterns = ts_model.find_temporal_patterns(
    dynamics,
    pattern_types=['waves', 'gradients', 'oscillations'],
    significance_test='permutation'
)
```

### Perturbation Prediction

```python
from spatial_omics_gfm.tasks import PerturbationPredictor

# Predict effects of perturbations
perturb_predictor = PerturbationPredictor(model)

# Single-cell perturbation
single_cell_effect = perturb_predictor.predict_single_cell_knockout(
    dataset,
    cell_id=42,
    gene='TNF',
    propagation_steps=5
)

# Regional perturbation
regional_effect = perturb_predictor.predict_regional_perturbation(
    dataset,
    region_mask=tumor_region,
    perturbation_type='cytokine_treatment',
    cytokine='IL2',
    concentration=100  # ng/ml
)

# Visualize perturbation effects
perturb_predictor.plot_perturbation_heatmap(
    regional_effect,
    genes_of_interest=['IFNG', 'PRF1', 'GZMB'],
    save_path='perturbation_effects.png'
)
```

## Visualization

### Interactive Spatial Viewer

```python
from spatial_omics_gfm.visualization import InteractiveSpatialViewer

viewer = InteractiveSpatialViewer()

# Create interactive plot
fig = viewer.create_interactive_plot(
    dataset,
    color_by='cell_type',
    size_by='total_counts',
    hover_data=['top_markers', 'pathway_scores']
)

# Add interaction overlay
viewer.add_interaction_layer(
    fig,
    interactions=predicted_interactions,
    show_top_n=50,
    edge_color_by='strength'
)

# Add custom annotations
viewer.add_annotations(
    fig,
    annotations={
        'regions': tissue_regions,
        'markers': landmark_cells
    }
)

# Save as interactive HTML
viewer.save(fig, 'interactive_spatial_map.html')
```

### Publication-Ready Figures

```python
from spatial_omics_gfm.visualization import PublicationPlotter

pub_plotter = PublicationPlotter(style='nature')

# Multi-panel figure
fig = pub_plotter.create_figure(
    n_rows=2,
    n_cols=3,
    figsize=(15, 10)
)

# Panel A: Spatial cell types
pub_plotter.plot_spatial_celltypes(
    fig.axes[0],
    dataset,
    cell_types=model_predictions,
    palette='spectral'
)

# Panel B: Interaction network
pub_plotter.plot_interaction_network(
    fig.axes[1],
    interactions,
    layout='circular',
    node_size_by='degree'
)

# Panel C: Pathway heatmap
pub_plotter.plot_pathway_heatmap(
    fig.axes[2],
    pathway_scores,
    cluster_rows=True,
    cmap='RdBu_r'
)

# Save publication figure
pub_plotter.save_figure(
    fig,
    'figure_3.pdf',
    dpi=300,
    bbox_inches='tight'
)
```

## Performance and Scaling

### Distributed Training

```python
from spatial_omics_gfm.training import DistributedTrainer
import torch.distributed as dist

# Multi-GPU training setup
trainer = DistributedTrainer(
    model=model,
    dataset=large_dataset,
    num_gpus=8,
    strategy='ddp'  # or 'fsdp' for very large models
)

# Train with gradient accumulation
trainer.train(
    epochs=100,
    batch_size_per_gpu=2,
    gradient_accumulation_steps=16,
    mixed_precision=True,
    checkpointing=True  # Gradient checkpointing
)

# Monitor training
trainer.log_metrics(
    wandb_project='spatial-gfm',
    log_frequency=100
)
```

### Efficient Inference

```python
from spatial_omics_gfm.inference import EfficientInference

# Setup for large-scale inference
inference_engine = EfficientInference(
    model=model,
    batch_size=32,
    use_amp=True,
    compile_model=True  # Torch 2.0 compile
)

# Process large dataset in chunks
results = inference_engine.process_large_dataset(
    dataset_path='huge_spatial_data.h5',
    chunk_size=10000,
    overlap=100,  # For boundary effects
    save_intermediate=True
)

# Memory-mapped inference for extremely large datasets
mmap_results = inference_engine.memory_mapped_inference(
    mmap_file='spatial_data.mmap',
    output_file='predictions.mmap',
    max_memory_gb=16
)
```

## Integration with Other Tools

### Scanpy Integration

```python
from spatial_omics_gfm.integrations import ScanpyBridge
import scanpy as sc

# Convert to/from AnnData
adata = sc.read_h5ad('spatial_data.h5ad')

# Apply GFM predictions to AnnData
bridge = ScanpyBridge()
adata = bridge.add_gfm_predictions(
    adata,
    model=model,
    predictions=['cell_types', 'interactions', 'pathways'],
    key_added='gfm'
)

# Use Scanpy functions with GFM results
sc.tl.umap(adata, use_rep='X_gfm')
sc.pl.umap(adata, color=['gfm_cell_types', 'gfm_confidence'])
```

### Squidpy Integration

```python
from spatial_omics_gfm.integrations import SquidpyBridge
import squidpy as sq

# Spatial analysis with GFM
squidpy_bridge = SquidpyBridge()

# Enhance spatial statistics with GFM
spatial_stats = squidpy_bridge.compute_enhanced_spatial_statistics(
    adata,
    model=model,
    statistics=['moran', 'geary', 'lee'],
    use_gfm_embeddings=True
)

# GFM-guided neighborhood enrichment
enrichment = squidpy_bridge.neighborhood_enrichment(
    adata,
    cluster_key='gfm_cell_types',
    method='gfm_aware'
)
```

## Benchmarking

### Standard Benchmarks

```python
from spatial_omics_gfm.benchmarks import SpatialOmicsBenchmark

benchmark = SpatialOmicsBenchmark()

# Run on standard datasets
results = benchmark.evaluate(
    model=model,
    datasets=[
        'mouse_brain_cortex',
        'human_lymph_node',
        'tumor_microenvironment',
        'developing_heart'
    ],
    metrics=[
        'cell_type_accuracy',
        'interaction_aupr',
        'spatial_coherence',
        'pathway_enrichment'
    ]
)

# Compare with baselines
benchmark.compare_methods(
    methods={
        'spatial-gfm': model,
        'graphsaint': baseline1,
        'stgcn': baseline2,
        'tangram': baseline3
    },
    save_results='benchmark_results.csv'
)
```

### Custom Metrics

```python
from spatial_omics_gfm.metrics import SpatialMetrics

metrics = SpatialMetrics()

# Spatial coherence of predictions
coherence = metrics.spatial_coherence(
    predictions=cell_type_predictions,
    spatial_graph=dataset.spatial_graph,
    method='moran_i'
)

# Interaction prediction quality
interaction_quality = metrics.interaction_metrics(
    predicted=predicted_interactions,
    ground_truth=known_interactions,
    spatial_context=True
)

# Biological plausibility
plausibility = metrics.biological_plausibility(
    predictions=model_outputs,
    known_biology=curated_knowledge,
    penalty_impossible=True
)
```

## Advanced Applications

### Spatial Cell State Dynamics

```python
from spatial_omics_gfm.applications import CellStateDynamics

dynamics_analyzer = CellStateDynamics(model)

# Infer cell state transitions
transitions = dynamics_analyzer.infer_transitions(
    dataset,
    method='optimal_transport',
    spatial_constraint=True
)

# Find transition hotspots
hotspots = dynamics_analyzer.find_transition_zones(
    transitions,
    min_density=5,
    statistical_test='spatial_scan'
)

# Predict future states
future_states = dynamics_analyzer.predict_future_states(
    current_data=dataset,
    time_delta=24,  # hours
    uncertainty_quantification=True
)
```

### Disease Microenvironment Analysis

```python
from spatial_omics_gfm.applications import DiseaseMicroenvironment

disease_analyzer = DiseaseMicroenvironment(
    model=model,
    disease='breast_cancer'
)

# Identify microenvironment niches
niches = disease_analyzer.identify_niches(
    dataset,
    min_size=20,
    homogeneity_threshold=0.8
)

# Characterize tumor-immune interactions
tumor_immune = disease_analyzer.analyze_tumor_immune_interface(
    dataset,
    tumor_mask=tumor_annotations,
    immune_phenotypes=['CD8_T', 'CD4_T', 'Macrophage', 'DC']
)

# Predict treatment response
response = disease_analyzer.predict_treatment_response(
    dataset,
    treatment='anti_pd1',
    biomarkers=['PD1', 'PDL1', 'CTLA4'],
    spatial_features=True
)
```

### Developmental Trajectories

```python
from spatial_omics_gfm.applications import DevelopmentalAnalysis

dev_analyzer = DevelopmentalAnalysis(model)

# Reconstruct developmental paths
dev_paths = dev_analyzer.reconstruct_lineages(
    dataset,
    start_population='stem_cells',
    spatial_constraints=True,
    branching_points=True
)

# Find morphogen gradients
gradients = dev_analyzer.detect_morphogen_gradients(
    dataset,
    candidate_morphogens=['WNT', 'BMP', 'SHH'],
    gradient_metrics=['steepness', 'range', 'robustness']
)

# Predict cell fate
fate_predictions = dev_analyzer.predict_cell_fate(
    dataset,
    time_horizon=48,  # hours
    stochastic=True,
    n_simulations=100
)
```

## Configuration

### Model Configuration

```yaml
# config/model_config.yaml
model:
  architecture: "spatial_graph_transformer"
  num_parameters: 1_300_000_000
  hidden_dim: 1024
  num_layers: 24
  num_heads: 16
  dropout: 0.1
  
spatial_encoding:
  method: "learned_positional"
  max_distance: 1000  # micrometers
  distance_bins: 50
  
graph_construction:
  method: "knn"
  k: 10
  distance_threshold: 200
  edge_features: ["distance", "direction", "shared_markers"]
  
training:
  batch_size: 4
  gradient_accumulation: 8
  learning_rate: 1e-4
  warmup_steps: 10000
  max_steps: 1000000
```

### Data Processing Configuration

```yaml
# config/data_config.yaml
preprocessing:
  normalization: "scran"
  log_transform: true
  highly_variable_genes: 3000
  
quality_control:
  min_genes_per_cell: 200
  max_genes_per_cell: 5000
  min_cells_per_gene: 10
  mitochondrial_threshold: 0.2
  
spatial_processing:
  smooth_expression: true
  smoothing_method: "gaussian"
  sigma: 1.5
  
augmentation:
  rotation: true
  flip: true
  expression_noise: 0.1
  dropout: 0.1
```

## Troubleshooting

### Common Issues

```python
from spatial_omics_gfm.diagnostics import DiagnosticTool

diagnostic = DiagnosticTool()

# Check data quality
data_report = diagnostic.check_data_quality(dataset)
if data_report.has_issues:
    # Apply automatic fixes
    dataset = diagnostic.auto_fix_data(dataset)

# Model convergence issues
if not model.is_converging():
    # Adjust learning rate
    diagnostic.suggest_learning_rate(model, dataset)
    
    # Check for data imbalance
    balance_report = diagnostic.check_class_balance(dataset)
    if balance_report.is_imbalanced:
        dataset = diagnostic.balance_dataset(dataset)

# Memory issues
if diagnostic.predict_memory_usage(model, dataset) > available_memory:
    # Enable memory optimizations
    model.enable_gradient_checkpointing()
    model.enable_mixed_precision()
    
    # Suggest batch size
    optimal_batch = diagnostic.suggest_batch_size(
        model, dataset, available_memory
    )
```

## Citation

```bibtex
@article{spatial_omics_gfm,
  title={Spatial-Omics GFM: A Graph Foundation Model for Spatial Transcriptomics},
  author={Your Name},
  journal={Nature Methods},
  year={2025},
  doi={10.1038/nmeth.2025.xxxxx}
}
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- 10X Genomics, Slide-seq, and Xenium teams for platform development
- Single cell and spatial omics communities
- Graph neural network researchers

## Resources

- [Documentation](https://spatial-omics-gfm.readthedocs.io)
- [Model Zoo](https://huggingface.co/spatial-omics-gfm)
- [Tutorials](https://github.com/yourusername/spatial-omics-gfm/tutorials)
- [Discussion Forum](https://discourse.spatial-omics-gfm.org)
