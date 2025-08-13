#!/usr/bin/env python3
"""
Quick Start Example for Spatial-Omics GFM

This example demonstrates the basic functionality of the Spatial-Omics GFM
library without requiring heavy dependencies like PyTorch.
"""

from spatial_omics_gfm.core import (
    SimpleSpatialData,
    SimpleCellTypePredictor,
    SimpleInteractionPredictor,
    create_demo_data,
    run_basic_analysis
)

def main():
    """Run quick start example."""
    print("🧬 Spatial-Omics GFM Quick Start")
    print("=" * 50)
    
    # Option 1: Run complete basic analysis
    print("\nOption 1: Complete Basic Analysis")
    print("-" * 30)
    results = run_basic_analysis()
    
    # Option 2: Step-by-step analysis
    print("\n\nOption 2: Step-by-Step Analysis")
    print("-" * 30)
    
    # Create data
    print("📊 Creating synthetic data...")
    data = create_demo_data(n_cells=100, n_genes=50)
    data.normalize_expression()
    
    # Get basic stats
    stats = data.get_summary_stats()
    print(f"   Data: {stats['n_cells']} cells, {stats['n_genes']} genes")
    
    # Predict cell types
    print("🔬 Predicting cell types...")
    predictor = SimpleCellTypePredictor()
    cell_types = predictor.predict_cell_types(data)
    
    # Find best cell type for each cell
    import numpy as np
    best_types = []
    for i in range(data.n_cells):
        scores = {ct: scores[i] for ct, scores in cell_types.items()}
        best_type = max(scores, key=scores.get)
        best_types.append(best_type)
    
    # Count cell types
    from collections import Counter
    type_counts = Counter(best_types)
    print("   Cell type distribution:")
    for cell_type, count in type_counts.items():
        print(f"     {cell_type}: {count} cells")
    
    # Predict interactions
    print("🔗 Predicting interactions...")
    interaction_predictor = SimpleInteractionPredictor()
    interactions = interaction_predictor.predict_interactions(data, max_distance=50)
    
    if interactions:
        print(f"   Found {len(interactions)} interactions")
        top_interaction = max(interactions, key=lambda x: x['interaction_score'])
        print(f"   Top interaction: {top_interaction['ligand']} -> {top_interaction['receptor']} "
              f"(score: {top_interaction['interaction_score']:.2f})")
    else:
        print("   No significant interactions found")
    
    print("\n✅ Quick start complete!")
    return data, cell_types, interactions

if __name__ == "__main__":
    main()