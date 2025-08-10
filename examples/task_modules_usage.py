"""
Example usage of task-specific modules for Spatial-Omics GFM.

This script demonstrates how to use the task modules for various
spatial transcriptomics analysis tasks.
"""

import torch
import numpy as np
from spatial_omics_gfm.tasks import (
    TaskConfig, 
    CellTypeClassifier, 
    HierarchicalCellTypeClassifier,
    InteractionPredictor,
    PathwayAnalyzer,
    TissueSegmenter
)

def example_cell_type_classification():
    """Example of using CellTypeClassifier."""
    print("=== Cell Type Classification Example ===")
    
    # Create configuration
    config = TaskConfig(hidden_dim=1024, num_classes=5, dropout=0.1)
    
    # Define cell types
    cell_types = ['T_cell', 'B_cell', 'Neuron', 'Astrocyte', 'Microglia']
    
    # Initialize classifier
    classifier = CellTypeClassifier(config=config, cell_type_names=cell_types)
    
    # Example forward pass (normally you'd use embeddings from foundation model)
    batch_size = 100
    embeddings = torch.randn(batch_size, config.hidden_dim)
    
    # Make predictions
    with torch.no_grad():
        outputs = classifier(embeddings)
        predictions = outputs['predictions']
        probabilities = outputs['probabilities']
    
    print(f"Input shape: {embeddings.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Example predictions: {predictions[:5]}")
    print(f"Cell types: {cell_types}")
    print()

def example_hierarchical_cell_typing():
    """Example of hierarchical cell type classification."""
    print("=== Hierarchical Cell Type Classification Example ===")
    
    config = TaskConfig(hidden_dim=1024, num_classes=3, dropout=0.1)
    
    # Define cell type hierarchy
    hierarchy = {
        'T_cell': {
            'CD4_T': ['Th1', 'Th2', 'Treg'],
            'CD8_T': ['Cytotoxic', 'Memory']
        },
        'B_cell': {
            'Naive_B': ['Follicular'],
            'Memory_B': ['Class_switched']
        }
    }
    
    # Initialize hierarchical classifier
    classifier = HierarchicalCellTypeClassifier(config=config, cell_type_hierarchy=hierarchy)
    
    # Example usage
    batch_size = 50
    embeddings = torch.randn(batch_size, config.hidden_dim)
    
    with torch.no_grad():
        outputs = classifier(embeddings)
    
    print(f"Hierarchy levels: {list(classifier.hierarchy_levels.keys())}")
    print(f"Number of classes per level: {[len(classes) for classes in classifier.hierarchy_levels.values()]}")
    print()

def example_interaction_prediction():
    """Example of cell-cell interaction prediction."""
    print("=== Interaction Prediction Example ===")
    
    config = TaskConfig(hidden_dim=1024, num_classes=3, dropout=0.1)
    
    # Initialize interaction predictor
    predictor = InteractionPredictor(config=config, interaction_database="cellphonedb")
    
    # Example graph data
    num_cells = 100
    embeddings = torch.randn(num_cells, config.hidden_dim)
    
    # Create example edge index (spatial graph)
    edge_index = torch.randint(0, num_cells, (2, 300))  # 300 edges
    edge_attr = torch.randn(300, 3)  # distance, angle, type features
    
    with torch.no_grad():
        outputs = predictor(embeddings, edge_index=edge_index, edge_attr=edge_attr)
        interaction_probs = outputs['interaction_probabilities']
    
    print(f"Number of L-R pairs in database: {len(predictor.lr_database)}")
    print(f"Interaction probabilities shape: {interaction_probs.shape}")
    print(f"Example interaction types: ['ligand_receptor', 'paracrine', 'juxtacrine']")
    print()

def example_pathway_analysis():
    """Example of pathway analysis."""
    print("=== Pathway Analysis Example ===")
    
    config = TaskConfig(hidden_dim=1024, num_classes=50, dropout=0.1)
    
    # Initialize pathway analyzer
    analyzer = PathwayAnalyzer(config=config, pathway_database="kegg")
    
    # Example forward pass
    num_cells = 200
    embeddings = torch.randn(num_cells, config.hidden_dim)
    
    with torch.no_grad():
        outputs = analyzer(embeddings)
        pathway_scores = outputs['pathway_scores']
    
    print(f"Number of pathways: {len(analyzer.pathways)}")
    print(f"Pathway scores shape: {pathway_scores.shape}")
    print(f"Example pathways: {list(analyzer.pathways.keys())[:3]}")
    print()

def example_tissue_segmentation():
    """Example of tissue segmentation."""
    print("=== Tissue Segmentation Example ===")
    
    config = TaskConfig(hidden_dim=1024, num_classes=8, dropout=0.1)
    
    # Initialize tissue segmenter
    segmenter = TissueSegmenter(config=config, segmentation_method="hierarchical")
    
    # Example forward pass
    num_cells = 500
    embeddings = torch.randn(num_cells, config.hidden_dim)
    
    with torch.no_grad():
        outputs = segmenter(embeddings)
        region_probs = outputs['region_probabilities']
        predictions = outputs['predictions']
    
    print(f"Number of tissue regions: {config.num_classes}")
    print(f"Region probabilities shape: {region_probs.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique regions found: {torch.unique(predictions).tolist()}")
    print()

def example_evaluation():
    """Example of model evaluation."""
    print("=== Model Evaluation Example ===")
    
    config = TaskConfig(hidden_dim=512, num_classes=3, dropout=0.1)
    
    # Create a simple classifier
    classifier = CellTypeClassifier(config=config, cell_type_names=['A', 'B', 'C'])
    
    # Generate example data
    batch_size = 100
    embeddings = torch.randn(batch_size, config.hidden_dim)
    targets = torch.randint(0, 3, (batch_size,))
    
    # Make predictions
    with torch.no_grad():
        predictions = classifier(embeddings)
    
    # Evaluate performance
    metrics = classifier.evaluate(predictions, targets, metrics=['accuracy', 'f1'])
    
    print(f"Evaluation metrics: {metrics}")
    print()

if __name__ == "__main__":
    print("Spatial-Omics GFM Task Modules Usage Examples")
    print("=" * 50)
    
    example_cell_type_classification()
    example_hierarchical_cell_typing()
    example_interaction_prediction()
    example_pathway_analysis()
    example_tissue_segmentation()
    example_evaluation()
    
    print("🎉 All examples completed successfully!")