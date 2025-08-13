"""
Simple working example to demonstrate basic functionality.

This example works without heavy dependencies to show the core concepts.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import json


class SimpleSpatialData:
    """Simplified spatial transcriptomics data representation."""
    
    def __init__(
        self,
        expression_matrix: np.ndarray,
        coordinates: np.ndarray,
        gene_names: Optional[List[str]] = None
    ):
        """
        Initialize simple spatial data.
        
        Args:
            expression_matrix: Cell x Gene expression matrix
            coordinates: Cell x 2 coordinate matrix (x, y)
            gene_names: Optional list of gene names
        """
        self.expression_matrix = expression_matrix
        self.coordinates = coordinates
        self.n_cells, self.n_genes = expression_matrix.shape
        self.gene_names = gene_names or [f"Gene_{i}" for i in range(self.n_genes)]
        
        # Validate data
        assert coordinates.shape[0] == self.n_cells, "Coordinate count must match cell count"
        assert coordinates.shape[1] == 2, "Coordinates must be 2D (x, y)"
    
    def normalize_expression(self) -> None:
        """Normalize expression data using simple log normalization."""
        # Add pseudocount and log transform
        self.expression_matrix = np.log1p(self.expression_matrix)
        
        # Scale by total counts per cell
        cell_totals = np.sum(self.expression_matrix, axis=1, keepdims=True)
        cell_totals[cell_totals == 0] = 1  # Avoid division by zero
        self.expression_matrix = self.expression_matrix / cell_totals * 10000
    
    def find_spatial_neighbors(self, k: int = 6) -> Dict[int, List[int]]:
        """Find k nearest spatial neighbors for each cell."""
        neighbors = {}
        
        for i in range(self.n_cells):
            # Calculate distances to all other cells
            distances = np.sqrt(
                np.sum((self.coordinates - self.coordinates[i]) ** 2, axis=1)
            )
            # Find k+1 nearest (including self), then exclude self
            nearest_indices = np.argsort(distances)[1:k+1]
            neighbors[i] = nearest_indices.tolist()
        
        return neighbors
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get basic summary statistics."""
        return {
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "mean_expression_per_cell": float(np.mean(np.sum(self.expression_matrix, axis=1))),
            "mean_expression_per_gene": float(np.mean(np.sum(self.expression_matrix, axis=0))),
            "spatial_extent": {
                "x_range": [float(np.min(self.coordinates[:, 0])), float(np.max(self.coordinates[:, 0]))],
                "y_range": [float(np.min(self.coordinates[:, 1])), float(np.max(self.coordinates[:, 1]))]
            }
        }


class SimpleCellTypePredictor:
    """Simple cell type prediction using expression patterns."""
    
    def __init__(self):
        self.marker_genes = {
            "T_cell": ["CD3D", "CD3E", "CD8A", "CD4"],
            "B_cell": ["CD19", "CD20", "MS4A1", "CD79A"],
            "Macrophage": ["CD68", "CD163", "CSF1R", "ADGRE1"],
            "Fibroblast": ["COL1A1", "COL3A1", "FN1", "VIM"],
            "Epithelial": ["EPCAM", "KRT18", "KRT19", "CDH1"]
        }
    
    def predict_cell_types(self, data: SimpleSpatialData) -> Dict[str, List[float]]:
        """
        Predict cell types based on marker gene expression.
        
        Returns:
            Dictionary mapping cell type names to confidence scores per cell
        """
        predictions = {}
        
        for cell_type, markers in self.marker_genes.items():
            # Find marker genes that exist in the data
            available_markers = [
                gene for gene in markers 
                if gene in data.gene_names
            ]
            
            if available_markers:
                # Get indices of available marker genes
                marker_indices = [
                    data.gene_names.index(gene) 
                    for gene in available_markers
                ]
                
                # Calculate mean expression of marker genes per cell
                marker_expression = np.mean(
                    data.expression_matrix[:, marker_indices], 
                    axis=1
                )
                
                # Normalize to [0, 1] range
                if np.max(marker_expression) > 0:
                    marker_expression = marker_expression / np.max(marker_expression)
                
                predictions[cell_type] = marker_expression.tolist()
            else:
                # No markers available, assign random low scores
                predictions[cell_type] = (np.random.random(data.n_cells) * 0.1).tolist()
        
        return predictions


class SimpleInteractionPredictor:
    """Simple cell-cell interaction prediction."""
    
    def __init__(self):
        self.ligand_receptor_pairs = [
            ("CXCL12", "CXCR4"),
            ("TNF", "TNFRSF1A"),
            ("IL1B", "IL1R1"),
            ("VEGFA", "KDR"),
            ("TGFB1", "TGFBR1")
        ]
    
    def predict_interactions(
        self, 
        data: SimpleSpatialData, 
        max_distance: float = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Predict cell-cell interactions based on ligand-receptor pairs.
        
        Args:
            data: Spatial data
            max_distance: Maximum distance for interaction consideration
            
        Returns:
            List of predicted interactions with metadata
        """
        interactions = []
        neighbors = data.find_spatial_neighbors(k=10)
        
        for cell_i in range(data.n_cells):
            for neighbor_j in neighbors.get(cell_i, []):
                # Calculate distance
                distance = np.sqrt(
                    np.sum((data.coordinates[cell_i] - data.coordinates[neighbor_j]) ** 2)
                )
                
                if distance <= max_distance:
                    # Check for ligand-receptor interactions
                    for ligand, receptor in self.ligand_receptor_pairs:
                        if ligand in data.gene_names and receptor in data.gene_names:
                            ligand_idx = data.gene_names.index(ligand)
                            receptor_idx = data.gene_names.index(receptor)
                            
                            ligand_expr = data.expression_matrix[cell_i, ligand_idx]
                            receptor_expr = data.expression_matrix[neighbor_j, receptor_idx]
                            
                            # Simple interaction score
                            interaction_score = ligand_expr * receptor_expr
                            
                            if interaction_score > 0.1:  # Threshold
                                interactions.append({
                                    "sender_cell": int(cell_i),
                                    "receiver_cell": int(neighbor_j),
                                    "ligand": ligand,
                                    "receptor": receptor,
                                    "interaction_score": float(interaction_score),
                                    "distance": float(distance)
                                })
        
        return interactions


def create_demo_data(n_cells: int = 1000, n_genes: int = 500) -> SimpleSpatialData:
    """Create synthetic spatial transcriptomics data for demonstration."""
    
    # Generate random coordinates
    coordinates = np.random.rand(n_cells, 2) * 1000  # 1000x1000 spatial area
    
    # Generate expression matrix with some structure
    expression_matrix = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes)).astype(float)
    
    # Add some spatial structure by making nearby cells more similar
    for i in range(n_cells):
        # Find nearby cells
        distances = np.sqrt(np.sum((coordinates - coordinates[i]) ** 2, axis=1))
        nearby_cells = np.where(distances < 100)[0]  # Within distance 100
        
        if len(nearby_cells) > 1:
            # Make nearby cells more similar by adding correlated noise
            shared_pattern = np.random.rand(n_genes) * 2
            for j in nearby_cells:
                expression_matrix[j] += shared_pattern * np.random.rand()
    
    # Create realistic gene names including some markers
    marker_genes = [
        "CD3D", "CD3E", "CD8A", "CD4",  # T cell markers
        "CD19", "CD20", "MS4A1", "CD79A",  # B cell markers
        "CD68", "CD163", "CSF1R", "ADGRE1",  # Macrophage markers
        "COL1A1", "COL3A1", "FN1", "VIM",  # Fibroblast markers
        "EPCAM", "KRT18", "KRT19", "CDH1",  # Epithelial markers
        "CXCL12", "CXCR4", "TNF", "TNFRSF1A", "IL1B", "IL1R1", "VEGFA", "KDR", "TGFB1", "TGFBR1"  # Interaction genes
    ]
    
    other_genes = [f"Gene_{i:04d}" for i in range(n_genes - len(marker_genes))]
    gene_names = marker_genes + other_genes
    
    return SimpleSpatialData(expression_matrix, coordinates, gene_names[:n_genes])


def run_basic_analysis():
    """Run a complete basic analysis to demonstrate functionality."""
    print("=== Spatial-Omics GFM Basic Analysis Demo ===")
    
    # Create demo data
    print("1. Creating synthetic spatial transcriptomics data...")
    data = create_demo_data(n_cells=500, n_genes=100)
    
    # Normalize data
    print("2. Normalizing expression data...")
    data.normalize_expression()
    
    # Get summary stats
    print("3. Computing summary statistics...")
    stats = data.get_summary_stats()
    print(f"   - Cells: {stats['n_cells']}")
    print(f"   - Genes: {stats['n_genes']}")
    print(f"   - Mean expression per cell: {stats['mean_expression_per_cell']:.2f}")
    print(f"   - Spatial extent: X={stats['spatial_extent']['x_range']}, Y={stats['spatial_extent']['y_range']}")
    
    # Predict cell types
    print("4. Predicting cell types...")
    cell_type_predictor = SimpleCellTypePredictor()
    cell_type_predictions = cell_type_predictor.predict_cell_types(data)
    
    for cell_type, scores in cell_type_predictions.items():
        mean_confidence = np.mean(scores)
        max_confidence = np.max(scores)
        print(f"   - {cell_type}: mean={mean_confidence:.3f}, max={max_confidence:.3f}")
    
    # Predict interactions
    print("5. Predicting cell-cell interactions...")
    interaction_predictor = SimpleInteractionPredictor()
    interactions = interaction_predictor.predict_interactions(data, max_distance=100.0)
    
    print(f"   - Found {len(interactions)} potential interactions")
    if interactions:
        # Show top 5 interactions
        interactions_sorted = sorted(interactions, key=lambda x: x['interaction_score'], reverse=True)
        print("   - Top interactions:")
        for i, interaction in enumerate(interactions_sorted[:5]):
            print(f"     {i+1}. {interaction['ligand']} -> {interaction['receptor']} "
                  f"(score: {interaction['interaction_score']:.3f}, "
                  f"distance: {interaction['distance']:.1f})")
    
    # Save results
    print("6. Saving analysis results...")
    results = {
        "summary_stats": stats,
        "cell_type_predictions": cell_type_predictions,
        "interactions": interactions[:100],  # Save top 100 interactions
        "analysis_metadata": {
            "n_cells_analyzed": data.n_cells,
            "n_genes_analyzed": data.n_genes,
            "analysis_type": "basic_demo"
        }
    }
    
    with open("/root/repo/basic_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("✅ Basic analysis complete! Results saved to basic_analysis_results.json")
    return results


if __name__ == "__main__":
    run_basic_analysis()