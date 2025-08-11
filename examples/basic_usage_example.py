#!/usr/bin/env python3
"""
Basic Usage Example for Spatial-Omics Graph Foundation Model

This script demonstrates how to use the Spatial-Omics GFM for basic spatial
transcriptomics analysis including:

1. Loading synthetic Visium-like data
2. Creating a Graph Foundation Model
3. Running cell type prediction
4. Basic visualization

Requirements:
- spatial-omics-gfm
- torch
- numpy
- pandas
- matplotlib
- scikit-learn
"""

import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from spatial_omics_gfm import SpatialGraphTransformer
from spatial_omics_gfm.models.graph_transformer import TransformerConfig, create_model_config
from spatial_omics_gfm.data.visium import VisiumDataset
from spatial_omics_gfm.data.base import SpatialDataConfig
from spatial_omics_gfm.tasks.cell_typing import CellTypeClassifier, CellTypeConfig
import scanpy as sc
import anndata as ad

def create_synthetic_visium_data(n_spots: int = 1000, n_genes: int = 2000) -> ad.AnnData:
    """
    Create synthetic Visium-like spatial transcriptomics data for testing.
    
    Args:
        n_spots: Number of spatial spots
        n_genes: Number of genes
        
    Returns:
        AnnData object with synthetic spatial transcriptomics data
    """
    print(f"Creating synthetic Visium data: {n_spots} spots, {n_genes} genes")
    
    # Create synthetic gene expression data
    # Simulate different cell types with distinct expression patterns
    np.random.seed(42)
    
    # Define cell types and their signatures
    cell_types = ['Epithelial', 'Immune', 'Stromal', 'Endothelial']
    n_cell_types = len(cell_types)
    
    # Assign spots to cell types
    spot_cell_types = np.random.choice(cell_types, size=n_spots)
    
    # Create expression matrix
    X = np.random.negative_binomial(n=5, p=0.3, size=(n_spots, n_genes)).astype(float)
    
    # Add cell type-specific signatures
    type_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    for spot_idx, cell_type in enumerate(spot_cell_types):
        type_idx = type_to_idx[cell_type]
        # Enhance expression of signature genes for this cell type
        signature_genes = range(type_idx * 50, (type_idx + 1) * 50)
        X[spot_idx, signature_genes] *= (3 + np.random.exponential(2))
    
    # Create spatial coordinates in a grid-like pattern with some noise
    grid_size = int(np.sqrt(n_spots))
    x_coords = []
    y_coords = []
    
    for i in range(n_spots):
        base_x = (i % grid_size) * 100
        base_y = (i // grid_size) * 100
        
        # Add noise to make it more realistic
        x_coords.append(base_x + np.random.normal(0, 15))
        y_coords.append(base_y + np.random.normal(0, 15))
    
    spatial_coords = np.column_stack([x_coords, y_coords])
    
    # Create gene names
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    
    # Create spot barcodes
    spot_barcodes = [f"SPOT_{i:06d}" for i in range(n_spots)]
    
    # Create AnnData object
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({
            'cell_type': spot_cell_types,
            'in_tissue': True,
            'total_counts': X.sum(axis=1),
            'n_genes': (X > 0).sum(axis=1)
        }, index=spot_barcodes),
        var=pd.DataFrame({
            'gene_ids': gene_names,
            'feature_types': 'Gene Expression'
        }, index=gene_names),
        obsm={
            'spatial': spatial_coords
        },
        uns={
            'spatial': {
                'scalefactors': {
                    'tissue_hires_scalef': 1.0,
                    'tissue_lowres_scalef': 1.0,
                    'fiducial_diameter_fullres': 1.0,
                    'spot_diameter_fullres': 65.0
                }
            }
        }
    )
    
    print(f"Created synthetic data with shape: {adata.shape}")
    return adata

def main():
    """Main example workflow."""
    print("=" * 60)
    print("Spatial-Omics GFM Basic Usage Example")
    print("=" * 60)
    
    # Step 1: Create synthetic data
    print("\n1. Creating synthetic Visium data...")
    adata = create_synthetic_visium_data(n_spots=500, n_genes=1000)
    
    # Step 2: Basic preprocessing
    print("\n2. Preprocessing data...")
    
    # Basic quality control and preprocessing
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    
    # Filter genes and spots
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_cells(adata, min_genes=100)
    
    # Normalize and log transform
    adata.raw = adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Select highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=500)
    adata = adata[:, adata.var.highly_variable].copy()
    
    print(f"After preprocessing: {adata.shape}")
    
    # Step 3: Build spatial graph
    print("\n3. Building spatial graph...")
    from spatial_omics_gfm.data.graph_construction import SpatialGraphBuilder
    
    graph_builder = SpatialGraphBuilder()
    edge_index, edge_attr, graph_info = graph_builder.build_spatial_graph(
        adata.obsm['spatial'], 
        method='knn',
        k=6
    )
    
    print(f"Built graph with {edge_index.shape[1]} edges")
    
    # Step 4: Create and configure the foundation model
    print("\n4. Setting up Spatial Graph Transformer...")
    
    # Create model configuration
    config = create_model_config(
        num_genes=adata.n_vars,
        model_size="base"
    )
    
    # Initialize the foundation model
    model = SpatialGraphTransformer(config)
    param_count = model.get_parameter_count()
    print(f"Model initialized with {param_count['total']:,} parameters")
    
    # Step 5: Test basic forward pass
    print("\n5. Testing model forward pass...")
    
    # Prepare inputs
    gene_expression = torch.tensor(adata.X.toarray(), dtype=torch.float32)
    spatial_coords = torch.tensor(adata.obsm['spatial'], dtype=torch.float32)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model.forward(
            gene_expression=gene_expression,
            spatial_coords=spatial_coords,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        embeddings = output['embeddings']
        print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Step 6: Set up cell type classification task
    print("\n6. Setting up cell type classification...")
    
    cell_types = adata.obs['cell_type'].unique().tolist()
    print(f"Cell types: {cell_types}")
    
    # Create cell type classifier configuration
    cell_type_config = CellTypeConfig(
        hidden_dim=config.hidden_dim,
        num_classes=len(cell_types),
        use_spatial_context=True,
        use_uncertainty=False
    )
    
    # Initialize classifier
    classifier = CellTypeClassifier(cell_type_config, cell_types)
    
    # Test prediction
    with torch.no_grad():
        predictions = classifier.forward(
            embeddings=embeddings,
            edge_index=edge_index,
            spatial_coords=spatial_coords
        )
        
        predicted_probs = predictions['probabilities']
        predicted_classes = predictions['predictions']
        
        print(f"Prediction probabilities shape: {predicted_probs.shape}")
        print(f"Predicted classes shape: {predicted_classes.shape}")
    
    # Step 7: Basic evaluation
    print("\n7. Basic evaluation...")
    
    # Convert true labels to indices
    label_to_idx = {label: idx for idx, label in enumerate(cell_types)}
    true_labels = torch.tensor([
        label_to_idx[label] for label in adata.obs['cell_type']
    ], dtype=torch.long)
    
    # Compute accuracy
    accuracy = (predicted_classes == true_labels).float().mean()
    print(f"Random prediction accuracy: {accuracy:.3f}")
    print("Note: This is random since the model is not trained!")
    
    # Step 8: Basic visualization
    print("\n8. Creating basic visualizations...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Spatial coordinates colored by true cell type
        ax1 = axes[0, 0]
        coords = adata.obsm['spatial']
        cell_type_colors = {'Epithelial': 'red', 'Immune': 'blue', 
                           'Stromal': 'green', 'Endothelial': 'orange'}
        
        for cell_type in cell_types:
            mask = adata.obs['cell_type'] == cell_type
            if mask.sum() > 0:
                ax1.scatter(coords[mask, 0], coords[mask, 1], 
                          c=cell_type_colors.get(cell_type, 'gray'),
                          label=cell_type, alpha=0.7, s=20)
        
        ax1.set_title('True Cell Types')
        ax1.set_xlabel('Spatial X')
        ax1.set_ylabel('Spatial Y')
        ax1.legend()
        
        # Plot 2: Predicted cell types
        ax2 = axes[0, 1]
        predicted_labels = [cell_types[idx.item()] for idx in predicted_classes]
        
        for cell_type in cell_types:
            mask = np.array(predicted_labels) == cell_type
            if mask.sum() > 0:
                ax2.scatter(coords[mask, 0], coords[mask, 1],
                          c=cell_type_colors.get(cell_type, 'gray'),
                          label=cell_type, alpha=0.7, s=20)
        
        ax2.set_title('Predicted Cell Types (Untrained)')
        ax2.set_xlabel('Spatial X')
        ax2.set_ylabel('Spatial Y')
        ax2.legend()
        
        # Plot 3: Total gene expression
        ax3 = axes[1, 0]
        total_expr = adata.obs['total_counts']
        scatter = ax3.scatter(coords[:, 0], coords[:, 1], c=total_expr, 
                            cmap='viridis', s=20, alpha=0.7)
        plt.colorbar(scatter, ax=ax3, label='Total Counts')
        ax3.set_title('Total Gene Expression')
        ax3.set_xlabel('Spatial X')
        ax3.set_ylabel('Spatial Y')
        
        # Plot 4: Number of genes detected
        ax4 = axes[1, 1]
        n_genes = adata.obs['n_genes']
        scatter = ax4.scatter(coords[:, 0], coords[:, 1], c=n_genes,
                            cmap='plasma', s=20, alpha=0.7)
        plt.colorbar(scatter, ax=ax4, label='N Genes')
        ax4.set_title('Number of Genes Detected')
        ax4.set_xlabel('Spatial X')
        ax4.set_ylabel('Spatial Y')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = project_root / "examples" / "basic_usage_plots.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plots saved to: {output_path}")
        
        # Show plot if in interactive environment
        plt.show()
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("This might be due to missing display or matplotlib backend issues")
    
    # Step 9: Summary
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nSummary:")
    print(f"- Processed {adata.n_obs} spatial spots with {adata.n_vars} genes")
    print(f"- Built spatial graph with {edge_index.shape[1]} edges")
    print(f"- Foundation model has {param_count['total']:,} parameters")
    print(f"- Generated embeddings of shape {embeddings.shape}")
    print(f"- Classified into {len(cell_types)} cell types")
    print(f"- Random accuracy: {accuracy:.3f} (expected since untrained)")
    
    print("\nNext steps:")
    print("- Train the foundation model on real spatial transcriptomics data")
    print("- Fine-tune task-specific heads for cell typing, interaction prediction, etc.")
    print("- Evaluate on validation datasets")
    print("- Apply to your own spatial transcriptomics data")
    
    print("\nFor more examples, check out:")
    print("- examples/visium_analysis_example.py")
    print("- examples/cell_interaction_example.py")
    print("- examples/training_example.py")

if __name__ == "__main__":
    main()