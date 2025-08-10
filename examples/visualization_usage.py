"""
Example usage of the Spatial-Omics GFM visualization modules.

This script demonstrates how to use the various visualization components
for spatial transcriptomics analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import visualization modules
from spatial_omics_gfm.visualization import (
    SpatialPlotter,
    InteractionNetworkPlotter,
    PathwayMapper,
    InteractiveSpatialViewer,
    PublicationPlotter
)

# Mock data classes for demonstration
class MockSpatialDataset:
    """Mock spatial dataset for demonstration."""
    
    def __init__(self, n_cells=1000, n_genes=2000):
        self.num_cells = n_cells
        self.num_genes = n_genes
        self.gene_names = [f"Gene_{i}" for i in range(n_genes)]
        
        # Generate mock spatial coordinates
        self.spatial_coords = np.random.uniform(0, 1000, (n_cells, 2))
        
        # Generate mock expression data
        self.expression_data = np.random.lognormal(0, 1, (n_cells, n_genes))
        
        # Generate mock cell types
        self.cell_types = np.random.choice(['T_Cell', 'B_Cell', 'NK_Cell', 'Monocyte', 'Fibroblast'], n_cells)
    
    def get(self, idx):
        """Get data for a specific sample."""
        class MockData:
            def __init__(self, coords, expression, cell_types):
                import torch
                self.pos = torch.tensor(coords, dtype=torch.float32)
                self.x = torch.tensor(expression, dtype=torch.float32)
                self.cell_type = cell_types
        
        return MockData(self.spatial_coords, self.expression_data, self.cell_types)


def generate_mock_predictions(n_cells=1000):
    """Generate mock model predictions."""
    return {
        'predictions': np.random.choice([0, 1, 2, 3, 4], n_cells),
        'confidence': np.random.uniform(0.3, 0.95, n_cells),
        'cell_type_names': ['T_Cell', 'B_Cell', 'NK_Cell', 'Monocyte', 'Fibroblast'],
        'total_cells': n_cells,
        'num_high_confidence': np.sum(np.random.uniform(0.3, 0.95, n_cells) > 0.8)
    }


def generate_mock_interactions(n_cells=1000, n_interactions=50):
    """Generate mock cell-cell interactions."""
    interactions = []
    for _ in range(n_interactions):
        cell_i = np.random.randint(0, n_cells)
        cell_j = np.random.randint(0, n_cells)
        if cell_i != cell_j:
            interactions.append({
                'cell_i': cell_i,
                'cell_j': cell_j,
                'confidence': np.random.uniform(0.5, 0.95)
            })
    
    return {
        'interactions': interactions,
        'num_interactions': len(interactions),
        'distance_threshold': 50.0
    }


def generate_mock_pathway_scores(n_pathways=20, n_cells=1000):
    """Generate mock pathway activity scores."""
    pathway_names = [f"Pathway_{i}" for i in range(n_pathways)]
    activity_matrix = np.random.uniform(0, 1, (n_pathways, n_cells))
    
    return {
        'pathway_activity': activity_matrix,
        'pathway_names': pathway_names,
        'pathways': {
            name: {
                'genes': [f"Gene_{i}" for i in range(10)],
                'interactions': [(f"Gene_{i}", f"Gene_{j}") for i in range(5) for j in range(i+1, 5)]
            } for name in pathway_names[:5]  # Only first 5 pathways have detailed info
        }
    }


def demonstrate_spatial_plotting():
    """Demonstrate spatial plotting capabilities."""
    print("=== Spatial Plotting Demo ===")
    
    # Create dataset and predictions
    dataset = MockSpatialDataset(n_cells=500)
    predictions = generate_mock_predictions(n_cells=500)
    interactions = generate_mock_interactions(n_cells=500, n_interactions=25)
    
    # Initialize plotter
    plotter = SpatialPlotter(style="publication", figsize=(12, 10))
    
    # Create spatial plot
    fig1 = plotter.plot_spatial(
        dataset=dataset,
        color_by="prediction",
        predictions=predictions,
        title="Spatial Cell Type Predictions",
        save_path="spatial_predictions.png"
    )
    plt.close(fig1)
    
    # Create gene expression plot
    fig2 = plotter.plot_gene_expression(
        dataset=dataset,
        gene_names=["Gene_0", "Gene_1", "Gene_10", "Gene_20"],
        title="Gene Expression Patterns",
        save_path="gene_expression.png"
    )
    plt.close(fig2)
    
    # Create interaction plot
    fig3 = plotter.plot_interactions(
        dataset=dataset,
        interactions=interactions,
        title="Cell-Cell Interactions",
        save_path="interactions.png"
    )
    plt.close(fig3)
    
    # Create summary plot
    fig4 = plotter.create_summary_plot(
        dataset=dataset,
        predictions=predictions,
        interactions=interactions,
        save_path="summary_plot.png"
    )
    plt.close(fig4)
    
    print("✓ Created spatial plots: spatial_predictions.png, gene_expression.png, interactions.png, summary_plot.png")


def demonstrate_interaction_networks():
    """Demonstrate interaction network visualization."""
    print("\n=== Interaction Network Demo ===")
    
    # Create dataset and interactions
    dataset = MockSpatialDataset(n_cells=300)
    interactions = generate_mock_interactions(n_cells=300, n_interactions=40)
    cell_types = ['T_Cell', 'B_Cell', 'NK_Cell', 'Monocyte'] * 75  # 300 cells
    
    # Initialize plotter
    network_plotter = InteractionNetworkPlotter(figsize=(14, 12))
    
    # Create network plot
    fig1 = network_plotter.plot_interaction_network(
        interactions=interactions,
        dataset=dataset,
        cell_types=cell_types,
        title="Cell-Cell Interaction Network",
        save_path="interaction_network.png"
    )
    plt.close(fig1)
    
    # Create spatial interaction plot
    fig2 = network_plotter.plot_spatial_interactions(
        dataset=dataset,
        interactions=interactions,
        title="Spatial Interactions",
        save_path="spatial_interactions.png"
    )
    plt.close(fig2)
    
    # Create interaction heatmap
    fig3 = network_plotter.plot_interaction_heatmap(
        interactions=interactions,
        cell_types=cell_types,
        title="Cell Type Interaction Heatmap",
        save_path="interaction_heatmap.png"
    )
    plt.close(fig3)
    
    print("✓ Created interaction plots: interaction_network.png, spatial_interactions.png, interaction_heatmap.png")


def demonstrate_pathway_visualization():
    """Demonstrate pathway visualization."""
    print("\n=== Pathway Visualization Demo ===")
    
    # Create pathway data
    dataset = MockSpatialDataset(n_cells=400)
    pathway_scores = generate_mock_pathway_scores(n_pathways=15, n_cells=400)
    
    # Initialize plotter
    pathway_plotter = PathwayMapper(figsize=(12, 10))
    
    # Create pathway heatmap
    fig1 = pathway_plotter.plot_pathway_heatmap(
        pathway_scores=pathway_scores,
        pathways=pathway_scores['pathway_names'][:10],
        title="Pathway Activity Heatmap",
        save_path="pathway_heatmap.png"
    )
    plt.close(fig1)
    
    # Create pathway network
    fig2 = pathway_plotter.plot_pathway_network(
        pathway_data=pathway_scores,
        pathway_name="Pathway_0",
        title="Pathway Network Diagram",
        save_path="pathway_network.png"
    )
    plt.close(fig2)
    
    # Create spatial pathway activity
    fig3 = pathway_plotter.plot_spatial_pathway_activity(
        dataset=dataset,
        pathway_scores=pathway_scores,
        pathway_name="Pathway_0",
        title="Spatial Pathway Activity",
        save_path="spatial_pathway.png"
    )
    plt.close(fig3)
    
    print("✓ Created pathway plots: pathway_heatmap.png, pathway_network.png, spatial_pathway.png")


def demonstrate_publication_figures():
    """Demonstrate publication-ready figure generation."""
    print("\n=== Publication Figure Demo ===")
    
    # Create data
    dataset = MockSpatialDataset(n_cells=600)
    predictions = generate_mock_predictions(n_cells=600)
    interactions = generate_mock_interactions(n_cells=600, n_interactions=30)
    pathway_scores = generate_mock_pathway_scores(n_pathways=12, n_cells=600)
    
    # Initialize plotter
    pub_plotter = PublicationPlotter(style="nature", dpi=300)
    
    # Create main figure
    fig1 = pub_plotter.create_figure_main(
        dataset=dataset,
        predictions=predictions,
        interactions=interactions,
        pathway_scores=pathway_scores,
        title="Spatial-Omics GFM Analysis",
        save_path="figure_main.pdf"
    )
    plt.close(fig1)
    
    # Create methods figure
    fig2 = pub_plotter.create_methods_figure(
        dataset=dataset,
        save_path="figure_methods.pdf"
    )
    plt.close(fig2)
    
    # Create comparison figure
    results_comparison = {
        'Method_A': {'accuracy': 0.75},
        'Method_B': {'accuracy': 0.82},
        'Our_Method': {'accuracy': 0.91}
    }
    
    fig3 = pub_plotter.create_comparison_figure(
        results_comparison=results_comparison,
        save_path="figure_comparison.pdf"
    )
    plt.close(fig3)
    
    print("✓ Created publication figures: figure_main.pdf, figure_methods.pdf, figure_comparison.pdf")


def demonstrate_interactive_visualization():
    """Demonstrate interactive visualization capabilities."""
    print("\n=== Interactive Visualization Demo ===")
    
    # Create data
    dataset = MockSpatialDataset(n_cells=400)
    predictions = generate_mock_predictions(n_cells=400)
    interactions = generate_mock_interactions(n_cells=400, n_interactions=25)
    pathway_scores = generate_mock_pathway_scores(n_pathways=10, n_cells=400)
    
    # Initialize interactive viewer
    viewer = InteractiveSpatialViewer(dataset=dataset)
    
    # Create interactive spatial plot
    fig1 = viewer.create_interactive_spatial_plot(
        dataset=dataset,
        predictions=predictions,
        color_by="prediction",
        title="Interactive Spatial Predictions"
    )
    
    # Save as HTML
    viewer.export_plot(fig1, "interactive_spatial.html", format="html")
    
    # Create interactive gene expression plot
    fig2 = viewer.create_interactive_gene_expression_plot(
        dataset=dataset,
        gene_names=["Gene_0", "Gene_5", "Gene_10", "Gene_15"],
        title="Interactive Gene Expression"
    )
    
    viewer.export_plot(fig2, "interactive_genes.html", format="html")
    
    # Create interactive interaction plot
    fig3 = viewer.create_interactive_interaction_plot(
        interactions=interactions,
        dataset=dataset,
        title="Interactive Cell Interactions"
    )
    
    viewer.export_plot(fig3, "interactive_interactions.html", format="html")
    
    print("✓ Created interactive plots: interactive_spatial.html, interactive_genes.html, interactive_interactions.html")
    
    # Note: Dashboard creation would require running the server
    print("Note: To create interactive dashboard, run:")
    print("app = viewer.create_dashboard(dataset, predictions, interactions, pathway_scores)")
    print("viewer.run_dashboard(app)")


def main():
    """Run all visualization demonstrations."""
    print("Spatial-Omics GFM Visualization Demo")
    print("====================================")
    
    # Create output directory
    output_dir = Path("visualization_output")
    output_dir.mkdir(exist_ok=True)
    
    # Change to output directory
    import os
    os.chdir(output_dir)
    
    try:
        # Run all demonstrations
        demonstrate_spatial_plotting()
        demonstrate_interaction_networks()
        demonstrate_pathway_visualization()
        demonstrate_publication_figures()
        demonstrate_interactive_visualization()
        
        print("\n🎉 All visualization demos completed successfully!")
        print(f"Output files saved to: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()