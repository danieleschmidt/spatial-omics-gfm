# Task-Specific Modules for Spatial-Omics GFM

This directory contains task-specific modules that build upon the SpatialGraphTransformer foundation model to perform various spatial transcriptomics analysis tasks.

## Overview

The task modules provide specialized heads and analysis pipelines for:

- **Cell Type Classification**: Predict cell types with hierarchical support
- **Cell-Cell Interaction Prediction**: Analyze ligand-receptor pairs and communication networks
- **Pathway Analysis**: Spatially-resolved pathway activity scoring and gradient analysis
- **Tissue Segmentation**: Automated tissue architecture analysis and region classification

## Architecture

All task modules inherit from the `BaseTask` class, which provides:

- Consistent interface for forward pass, loss computation, and evaluation
- Integration with the SpatialGraphTransformer foundation model
- Utilities for embedding extraction and model management
- Support for uncertainty quantification and confidence estimation

## Quick Start

```python
from spatial_omics_gfm.tasks import (
    TaskConfig, 
    CellTypeClassifier,
    InteractionPredictor,
    PathwayAnalyzer,
    TissueSegmenter
)
import torch

# Create configuration
config = TaskConfig(hidden_dim=1024, num_classes=10, dropout=0.1)

# Initialize task modules
cell_classifier = CellTypeClassifier(config=config, cell_type_names=['T_cell', 'B_cell', 'Neuron'])
interaction_predictor = InteractionPredictor(config=config)
pathway_analyzer = PathwayAnalyzer(config=config, pathway_database="kegg")
tissue_segmenter = TissueSegmenter(config=config)

# Example forward pass
embeddings = torch.randn(100, config.hidden_dim)
cell_predictions = cell_classifier(embeddings)
```

## Modules

### BaseTask (`base.py`)

The foundation class that all task modules inherit from.

**Key Features:**
- Abstract interface for `forward()` and `compute_loss()`
- Embedding extraction from foundation models or pre-computed features
- Model evaluation with standard metrics
- Parameter counting and model management

**Main Components:**
- `BaseTask`: Abstract base class
- `TaskConfig`: Configuration dataclass
- `ClassificationHead`: Multi-layer classification head
- `RegressionHead`: Multi-layer regression head
- `AttentionHead`: Attention mechanism for feature importance
- `MultiTaskHead`: Joint prediction across multiple tasks
- `UncertaintyHead`: Uncertainty estimation using evidential learning

### Cell Type Classification (`cell_typing.py`)

Predicts cell types from spatial transcriptomics data with spatial context awareness.

**Features:**
- Standard cell type classification with spatial context
- Hierarchical cell type classification respecting taxonomy
- Uncertainty quantification and confidence estimation
- Spatial neighborhood context encoding

**Classes:**
- `CellTypeClassifier`: Standard classifier with spatial context
- `HierarchicalCellTypeClassifier`: Multi-level hierarchical classification
- `SpatialContextEncoder`: Encodes spatial neighborhood information

**Usage:**
```python
# Standard classification
classifier = CellTypeClassifier(config, cell_type_names=['T_cell', 'B_cell'])
predictions = classifier.predict_from_adata(adata, foundation_model)

# Hierarchical classification
hierarchy = {'T_cell': {'CD4': ['Th1', 'Th2'], 'CD8': ['Cytotoxic']}}
hierarchical_classifier = HierarchicalCellTypeClassifier(config, hierarchy)
results = hierarchical_classifier.predict_from_adata(adata, foundation_model)
```

### Cell-Cell Interaction Prediction (`interaction_prediction.py`)

Analyzes cell-cell interactions including ligand-receptor pairs and communication networks.

**Features:**
- Ligand-receptor database integration (CellPhoneDB, NicheNet)
- Interaction type classification (paracrine, juxtacrine)
- Statistical significance testing via permutation
- Spatial communication network reconstruction
- Signaling neighborhood identification

**Classes:**
- `InteractionPredictor`: General interaction prediction
- `LigandReceptorPredictor`: Specialized L-R analysis
- `InteractionPredictionHead`: Neural network for interaction classification

**Usage:**
```python
predictor = InteractionPredictor(config, interaction_database="cellphonedb")
results = predictor.predict(adata, foundation_model, compute_significance=True)

# Access results
lr_interactions = results['ligand_receptor_analysis']
communication_networks = results['communication_networks']
significance = results['significance_results']
```

### Pathway Analysis (`pathway_analysis.py`)

Spatially-resolved pathway activity analysis with gradient detection.

**Features:**
- Multiple pathway databases (KEGG, Reactome, GO, Hallmark)
- Spatial smoothing of pathway activities
- Pathway gradient analysis and boundary detection
- Co-regulation network identification
- Multi-scale pathway zone identification

**Classes:**
- `PathwayAnalyzer`: General pathway analysis
- `SpatialPathwayAnalyzer`: Specialized spatial analysis
- `PathwayScoringHead`: Neural network for pathway scoring

**Usage:**
```python
analyzer = PathwayAnalyzer(config, pathway_database="kegg", spatial_smoothing=True)
results = analyzer.predict(adata, foundation_model, compute_gradients=True)

# Access results
pathway_scores = results['pathway_scores']
gradients = results['pathway_gradients']
zones = results['pathway_zones']
```

### Tissue Segmentation (`tissue_segmentation.py`)

Automated tissue architecture analysis and anatomical region identification.

**Features:**
- Multiple segmentation methods (hierarchical, spectral, watershed)
- Multi-scale segmentation analysis
- Region boundary detection and smoothing
- Region property computation (area, compactness)
- Anatomical region classification

**Classes:**
- `TissueSegmenter`: General tissue segmentation
- `RegionClassifier`: Specialized anatomical classification
- `TissueSegmentationHead`: Neural network for segmentation
- `RegionClassificationHead`: Neural network for region typing

**Usage:**
```python
segmenter = TissueSegmenter(config, segmentation_method="hierarchical")
results = segmenter.predict(adata, foundation_model, return_boundaries=True)

# Access results
region_assignments = results['region_assignments']
boundaries = results['region_boundaries']
properties = results['region_properties']
```

## Configuration

All task modules use the `TaskConfig` dataclass for configuration:

```python
@dataclass
class TaskConfig:
    hidden_dim: int = 1024          # Hidden dimension size
    num_classes: int = 10           # Number of output classes
    dropout: float = 0.1            # Dropout rate
    use_batch_norm: bool = True     # Use batch normalization
    activation: str = "gelu"        # Activation function
    confidence_threshold: float = 0.5  # Confidence threshold
```

## Integration with Foundation Model

Task modules can work with embeddings from the foundation model or pre-computed features:

```python
# Using foundation model directly
from spatial_omics_gfm.models import SpatialGraphTransformer

foundation_model = SpatialGraphTransformer(config)
classifier = CellTypeClassifier(task_config, cell_type_names)

# Method 1: Let task module extract embeddings
predictions = classifier.predict_from_adata(adata, foundation_model)

# Method 2: Pre-compute embeddings
embeddings = foundation_model(adata_to_tensors(adata))
predictions = classifier(embeddings)

# Method 3: Use pre-stored embeddings
adata.obsm['X_spatial_gfm'] = pre_computed_embeddings
predictions = classifier.predict_from_adata(adata)
```

## Evaluation and Metrics

All task modules support evaluation with standard metrics:

```python
# Evaluate predictions
metrics = classifier.evaluate(
    predictions, 
    ground_truth_labels, 
    metrics=['accuracy', 'precision', 'recall', 'f1']
)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 Score: {metrics['f1']:.3f}")
```

## Uncertainty Quantification

Task modules support uncertainty estimation:

```python
# Using evidential learning
config = TaskConfig(use_uncertainty=True, uncertainty_method="evidential")
classifier = CellTypeClassifier(config, cell_type_names)

predictions = classifier.predict_with_confidence(
    embeddings, 
    confidence_threshold=0.7
)

high_confidence_predictions = predictions['high_confidence_mask']
uncertainty_scores = predictions['uncertainty']
```

## Visualization

Task modules include visualization utilities:

```python
# Plot pathway activities
analyzer.plot_pathway_activity(adata, pathway_scores, pathways=['GLYCOLYSIS', 'APOPTOSIS'])

# Plot interaction networks
predictor.plot_interaction_network(adata, interactions, max_interactions=500)

# Plot tissue segmentation
segmenter.plot_segmentation(adata, region_assignments, boundaries)
```

## Examples

See `examples/task_modules_usage.py` for comprehensive usage examples.

## Contributing

When adding new task modules:

1. Inherit from `BaseTask`
2. Implement abstract methods: `forward()` and `compute_loss()`
3. Add prediction method for AnnData integration
4. Include evaluation and visualization utilities
5. Update `__init__.py` with new exports
6. Add comprehensive docstrings and examples