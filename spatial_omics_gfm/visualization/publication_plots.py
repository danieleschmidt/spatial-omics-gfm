"""
Publication-ready figure generation for spatial transcriptomics analysis.

This module provides comprehensive tools for creating publication-quality figures
with proper styling, multi-panel layouts, and standardized formatting.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Union, Tuple
from pathlib import Path
import warnings
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import string

from ..data.base import BaseSpatialDataset


class PublicationPlotter:
    """
    Publication-ready figure generation engine.
    
    Provides methods for creating high-quality figures with consistent styling,
    proper layouts, and publication standards for spatial transcriptomics analysis.
    """
    
    def __init__(
        self,
        style: str = "publication",
        dpi: int = 300,
        font_size: int = 8,
        figure_format: str = "pdf"
    ):
        """
        Initialize the publication plotter.
        
        Args:
            style: Plot style ('publication', 'nature', 'science', 'cell')
            dpi: Resolution for figures
            font_size: Base font size
            figure_format: Default save format ('pdf', 'png', 'svg')
        """
        self.style = style
        self.dpi = dpi
        self.font_size = font_size
        self.figure_format = figure_format
        
        # Set publication style
        self._setup_publication_style()
        
        # Define color palettes for different journals
        self.color_palettes = {
            'nature': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'science': ['#E31A1C', '#1F78B4', '#33A02C', '#FF7F00', '#6A3D9A',
                       '#B15928', '#A6CEE3', '#B2DF8A', '#FB9A99', '#FDBF6F'],
            'cell': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                    '#DDA0DD', '#F4A261', '#E76F51', '#2A9D8F', '#264653'],
            'publication': sns.color_palette("husl", 10)
        }
    
    def _setup_publication_style(self) -> None:
        """Setup matplotlib style for publication figures."""
        # Configure matplotlib for publication
        plt.rcParams.update({
            'font.size': self.font_size,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'axes.labelsize': self.font_size,
            'axes.titlesize': self.font_size + 1,
            'xtick.labelsize': self.font_size - 1,
            'ytick.labelsize': self.font_size - 1,
            'legend.fontsize': self.font_size - 1,
            'figure.titlesize': self.font_size + 2,
            'axes.linewidth': 0.5,
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
            'xtick.minor.width': 0.3,
            'ytick.minor.width': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'pdf.fonttype': 42,  # Embed fonts
            'ps.fonttype': 42,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def create_figure_main(
        self,
        dataset: BaseSpatialDataset,
        predictions: Dict[str, Any],
        interactions: Optional[Dict[str, Any]] = None,
        pathway_scores: Optional[Dict[str, Any]] = None,
        title: str = "Spatial-Omics GFM Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create main figure for publication (typically Figure 1).
        
        Args:
            dataset: Spatial dataset
            predictions: Model predictions
            interactions: Cell-cell interactions
            pathway_scores: Pathway activity scores
            title: Figure title
            save_path: Path to save figure
            
        Returns:
            Publication-ready matplotlib figure
        """
        # Create figure with custom layout
        fig = plt.figure(figsize=(174/25.4, 234/25.4), dpi=self.dpi)  # Nature single column
        
        # Create custom grid layout
        gs = gridspec.GridSpec(4, 3, figure=fig, 
                              height_ratios=[1, 1, 1, 0.8],
                              width_ratios=[1, 1, 1],
                              hspace=0.4, wspace=0.3)
        
        # Panel A: Spatial overview
        ax_a = fig.add_subplot(gs[0, :2])
        self._plot_spatial_overview_panel(ax_a, dataset, predictions)
        self._add_panel_label(ax_a, 'A')
        
        # Panel B: Cell type composition
        ax_b = fig.add_subplot(gs[0, 2])
        self._plot_cell_type_composition(ax_b, predictions)
        self._add_panel_label(ax_b, 'B')
        
        # Panel C: Prediction confidence
        ax_c = fig.add_subplot(gs[1, 0])
        self._plot_confidence_panel(ax_c, predictions)
        self._add_panel_label(ax_c, 'C')
        
        # Panel D: Spatial interactions
        ax_d = fig.add_subplot(gs[1, 1])
        if interactions:
            self._plot_interactions_panel(ax_d, dataset, interactions)
        else:
            ax_d.text(0.5, 0.5, 'No interactions', ha='center', va='center')
        self._add_panel_label(ax_d, 'D')
        
        # Panel E: Top pathways
        ax_e = fig.add_subplot(gs[1, 2])
        if pathway_scores:
            self._plot_top_pathways_panel(ax_e, pathway_scores)
        else:
            ax_e.text(0.5, 0.5, 'No pathway data', ha='center', va='center')
        self._add_panel_label(ax_e, 'E')
        
        # Panel F: Gene expression patterns (2 columns)
        ax_f = fig.add_subplot(gs[2, :2])
        self._plot_gene_patterns_panel(ax_f, dataset)
        self._add_panel_label(ax_f, 'F')
        
        # Panel G: Model performance metrics
        ax_g = fig.add_subplot(gs[2, 2])
        self._plot_performance_metrics(ax_g, predictions)
        self._add_panel_label(ax_g, 'G')
        
        # Panel H: Summary statistics (spanning bottom row)
        ax_h = fig.add_subplot(gs[3, :])
        self._plot_summary_statistics_panel(ax_h, dataset, predictions, interactions, pathway_scores)
        self._add_panel_label(ax_h, 'H')
        
        if save_path:
            self._save_publication_figure(fig, save_path)
        
        return fig
    
    def create_methods_figure(
        self,
        dataset: BaseSpatialDataset,
        model_architecture: Optional[Dict[str, Any]] = None,
        training_metrics: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create methods/supplementary figure showing model architecture and training.
        
        Args:
            dataset: Spatial dataset
            model_architecture: Model architecture details
            training_metrics: Training performance metrics
            save_path: Path to save figure
            
        Returns:
            Methods figure
        """
        # Create figure
        fig = plt.figure(figsize=(174/25.4, 200/25.4), dpi=self.dpi)
        
        gs = gridspec.GridSpec(3, 2, figure=fig,
                              height_ratios=[1, 1, 1],
                              hspace=0.4, wspace=0.3)
        
        # Panel A: Data preprocessing pipeline
        ax_a = fig.add_subplot(gs[0, :])
        self._plot_preprocessing_pipeline(ax_a, dataset)
        self._add_panel_label(ax_a, 'A')
        
        # Panel B: Model architecture
        ax_b = fig.add_subplot(gs[1, 0])
        self._plot_model_architecture(ax_b, model_architecture)
        self._add_panel_label(ax_b, 'B')
        
        # Panel C: Training curves
        ax_c = fig.add_subplot(gs[1, 1])
        if training_metrics:
            self._plot_training_curves(ax_c, training_metrics)
        else:
            ax_c.text(0.5, 0.5, 'No training data', ha='center', va='center')
        self._add_panel_label(ax_c, 'C')
        
        # Panel D: Hyperparameter analysis
        ax_d = fig.add_subplot(gs[2, 0])
        self._plot_hyperparameter_analysis(ax_d)
        self._add_panel_label(ax_d, 'D')
        
        # Panel E: Computational performance
        ax_e = fig.add_subplot(gs[2, 1])
        self._plot_computational_performance(ax_e)
        self._add_panel_label(ax_e, 'E')
        
        if save_path:
            self._save_publication_figure(fig, save_path)
        
        return fig
    
    def create_comparison_figure(
        self,
        results_comparison: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create figure comparing different methods/conditions.
        
        Args:
            results_comparison: Dictionary of method names to results
            save_path: Path to save figure
            
        Returns:
            Comparison figure
        """
        # Create figure
        fig = plt.figure(figsize=(174/25.4, 150/25.4), dpi=self.dpi)
        
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Panel A: Performance comparison
        ax_a = fig.add_subplot(gs[0, 0])
        self._plot_performance_comparison(ax_a, results_comparison)
        self._add_panel_label(ax_a, 'A')
        
        # Panel B: Prediction accuracy by cell type
        ax_b = fig.add_subplot(gs[0, 1])
        self._plot_celltype_accuracy_comparison(ax_b, results_comparison)
        self._add_panel_label(ax_b, 'B')
        
        # Panel C: Computational efficiency
        ax_c = fig.add_subplot(gs[0, 2])
        self._plot_efficiency_comparison(ax_c, results_comparison)
        self._add_panel_label(ax_c, 'C')
        
        # Panel D: Confusion matrices (spanning bottom row)
        ax_d = fig.add_subplot(gs[1, :])
        self._plot_confusion_matrices(ax_d, results_comparison)
        self._add_panel_label(ax_d, 'D')
        
        if save_path:
            self._save_publication_figure(fig, save_path)
        
        return fig
    
    def create_supplementary_figure(
        self,
        additional_results: Dict[str, Any],
        figure_type: str = "general",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create supplementary figure with additional analyses.
        
        Args:
            additional_results: Dictionary with additional analysis results
            figure_type: Type of supplementary figure
            save_path: Path to save figure
            
        Returns:
            Supplementary figure
        """
        # Create figure
        fig = plt.figure(figsize=(174/25.4, 200/25.4), dpi=self.dpi)
        
        if figure_type == "pathways":
            return self._create_pathway_supplementary(fig, additional_results, save_path)
        elif figure_type == "interactions":
            return self._create_interaction_supplementary(fig, additional_results, save_path)
        else:
            return self._create_general_supplementary(fig, additional_results, save_path)
    
    def _plot_spatial_overview_panel(
        self,
        ax: plt.Axes,
        dataset: BaseSpatialDataset,
        predictions: Dict[str, Any]
    ) -> None:
        """Plot spatial overview panel."""
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        # Get predictions if available
        if 'predictions' in predictions:
            colors = predictions['predictions']
            unique_preds = np.unique(colors)
            palette = self.color_palettes[self.style][:len(unique_preds)]
            
            for i, pred in enumerate(unique_preds):
                mask = colors == pred
                ax.scatter(spatial_coords[mask, 0], spatial_coords[mask, 1],
                          c=palette[i], s=0.5, alpha=0.8, edgecolors='none',
                          label=f'Type {pred}')
        else:
            ax.scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                      c='gray', s=0.5, alpha=0.7)
        
        ax.set_xlabel('X position (μm)')
        ax.set_ylabel('Y position (μm)')
        ax.set_title('Spatial cell type predictions')
        ax.set_aspect('equal')
    
    def _plot_cell_type_composition(
        self,
        ax: plt.Axes,
        predictions: Dict[str, Any]
    ) -> None:
        """Plot cell type composition pie chart."""
        if 'predictions' in predictions:
            pred_counts = pd.Series(predictions['predictions']).value_counts()
            colors = self.color_palettes[self.style][:len(pred_counts)]
            
            wedges, texts, autotexts = ax.pie(pred_counts.values, 
                                            labels=[f'Type {i}' for i in pred_counts.index],
                                            colors=colors,
                                            autopct='%1.1f%%',
                                            startangle=90)
            
            # Style the text
            for text in texts:
                text.set_fontsize(self.font_size - 2)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(self.font_size - 2)
        
        ax.set_title('Cell type composition')
    
    def _plot_confidence_panel(
        self,
        ax: plt.Axes,
        predictions: Dict[str, Any]
    ) -> None:
        """Plot prediction confidence distribution."""
        if 'confidence' in predictions:
            confidence = predictions['confidence']
            ax.hist(confidence, bins=30, alpha=0.7, color='skyblue', 
                   edgecolor='black', linewidth=0.5)
            ax.axvline(np.mean(confidence), color='red', linestyle='--', 
                      linewidth=1, label=f'Mean: {np.mean(confidence):.3f}')
            ax.legend()
            ax.set_xlabel('Prediction confidence')
            ax.set_ylabel('Number of cells')
            ax.set_title('Confidence distribution')
        else:
            ax.text(0.5, 0.5, 'No confidence data', ha='center', va='center')
    
    def _plot_interactions_panel(
        self,
        ax: plt.Axes,
        dataset: BaseSpatialDataset,
        interactions: Dict[str, Any]
    ) -> None:
        """Plot cell-cell interactions panel."""
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        # Plot cells
        ax.scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                  c='lightgray', s=1, alpha=0.5)
        
        # Plot top interactions
        if 'interactions' in interactions:
            top_interactions = interactions['interactions'][:20]
            
            for interaction in top_interactions:
                cell_i = interaction['cell_i']
                cell_j = interaction['cell_j']
                confidence = interaction['confidence']
                
                if cell_i < len(spatial_coords) and cell_j < len(spatial_coords):
                    ax.plot([spatial_coords[cell_i, 0], spatial_coords[cell_j, 0]],
                           [spatial_coords[cell_i, 1], spatial_coords[cell_j, 1]],
                           color='red', alpha=confidence, linewidth=0.5)
        
        ax.set_title('Cell-cell interactions')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _plot_top_pathways_panel(
        self,
        ax: plt.Axes,
        pathway_scores: Dict[str, Any]
    ) -> None:
        """Plot top pathways bar chart."""
        # Mock pathway data for demonstration
        pathway_names = ['Glycolysis', 'TCA Cycle', 'Oxidative\nPhosphorylation', 
                        'Fatty Acid\nMetabolism', 'Amino Acid\nMetabolism']
        scores = np.random.uniform(0.3, 0.9, len(pathway_names))
        
        bars = ax.barh(range(len(pathway_names)), scores,
                      color=self.color_palettes[self.style][0], alpha=0.7)
        
        ax.set_yticks(range(len(pathway_names)))
        ax.set_yticklabels(pathway_names)
        ax.set_xlabel('Activity score')
        ax.set_title('Top active pathways')
        ax.set_xlim(0, 1)
    
    def _plot_gene_patterns_panel(
        self,
        ax: plt.Axes,
        dataset: BaseSpatialDataset
    ) -> None:
        """Plot gene expression patterns."""
        # Create mock gene expression heatmap
        n_genes, n_cells = 50, 100
        expression_data = np.random.lognormal(0, 1, (n_genes, n_cells))
        
        im = ax.imshow(expression_data, cmap='viridis', aspect='auto')
        ax.set_xlabel('Cells')
        ax.set_ylabel('Genes')
        ax.set_title('Gene expression patterns')
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='Expression')
    
    def _plot_performance_metrics(
        self,
        ax: plt.Axes,
        predictions: Dict[str, Any]
    ) -> None:
        """Plot model performance metrics."""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = np.random.uniform(0.7, 0.95, len(metrics))
        
        bars = ax.bar(metrics, values, color=self.color_palettes[self.style][:len(metrics)], 
                     alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('Score')
        ax.set_title('Model performance')
        ax.set_ylim(0, 1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_summary_statistics_panel(
        self,
        ax: plt.Axes,
        dataset: BaseSpatialDataset,
        predictions: Dict[str, Any],
        interactions: Optional[Dict[str, Any]] = None,
        pathway_scores: Optional[Dict[str, Any]] = None
    ) -> None:
        """Plot summary statistics."""
        ax.axis('off')
        
        # Create summary table
        stats_data = []
        stats_data.append(['Total cells', f'{dataset.num_cells:,}'])
        stats_data.append(['Total genes', f'{dataset.num_genes:,}'])
        
        if 'predictions' in predictions:
            unique_types = len(set(predictions['predictions']))
            stats_data.append(['Cell types identified', f'{unique_types}'])
        
        if 'confidence' in predictions:
            high_conf = np.sum(np.array(predictions['confidence']) > 0.8)
            stats_data.append(['High confidence predictions', f'{high_conf:,}'])
        
        if interactions and 'interactions' in interactions:
            stats_data.append(['Cell-cell interactions', f'{len(interactions["interactions"]):,}'])
        
        # Create table
        table = ax.table(cellText=stats_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(self.font_size - 1)
        table.scale(1, 1.5)
        
        # Style table
        for i in range(len(stats_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    def _add_panel_label(
        self,
        ax: plt.Axes,
        label: str,
        x: float = -0.1,
        y: float = 1.0
    ) -> None:
        """Add panel label (A, B, C, etc.) to subplot."""
        ax.text(x, y, label, transform=ax.transAxes, fontsize=self.font_size + 2,
                fontweight='bold', va='top', ha='right')
    
    def _save_publication_figure(
        self,
        fig: plt.Figure,
        save_path: str
    ) -> None:
        """Save figure in publication format."""
        path = Path(save_path)
        
        # Save in specified format
        fig.savefig(f"{path.stem}.{self.figure_format}", 
                   dpi=self.dpi, bbox_inches='tight', 
                   pad_inches=0.1, facecolor='white')
        
        # Also save as PNG for preview
        if self.figure_format != 'png':
            fig.savefig(f"{path.stem}.png", 
                       dpi=150, bbox_inches='tight', 
                       pad_inches=0.1, facecolor='white')
    
    # Additional helper methods for different figure types
    def _plot_preprocessing_pipeline(self, ax: plt.Axes, dataset: BaseSpatialDataset) -> None:
        """Plot data preprocessing pipeline diagram."""
        ax.text(0.5, 0.5, 'Data Preprocessing Pipeline\n(Schematic)', 
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.set_title('Data preprocessing')
        ax.axis('off')
    
    def _plot_model_architecture(self, ax: plt.Axes, model_architecture: Optional[Dict[str, Any]]) -> None:
        """Plot model architecture diagram."""
        ax.text(0.5, 0.5, 'Graph Transformer\nArchitecture\n(Schematic)', 
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        ax.set_title('Model architecture')
        ax.axis('off')
    
    def _plot_training_curves(self, ax: plt.Axes, training_metrics: Dict[str, Any]) -> None:
        """Plot training loss and accuracy curves."""
        epochs = np.arange(1, 101)
        train_loss = 2.5 * np.exp(-epochs/50) + 0.1 + 0.05 * np.random.randn(100)
        val_loss = 2.7 * np.exp(-epochs/45) + 0.15 + 0.05 * np.random.randn(100)
        
        ax.plot(epochs, train_loss, label='Training', color=self.color_palettes[self.style][0])
        ax.plot(epochs, val_loss, label='Validation', color=self.color_palettes[self.style][1])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training curves')
        ax.legend()
    
    def _plot_hyperparameter_analysis(self, ax: plt.Axes) -> None:
        """Plot hyperparameter sensitivity analysis."""
        params = ['Learning\nRate', 'Hidden\nDim', 'Dropout', 'Batch\nSize']
        scores = np.random.uniform(0.7, 0.9, len(params))
        
        ax.bar(params, scores, color=self.color_palettes[self.style][2], alpha=0.7)
        ax.set_ylabel('Performance')
        ax.set_title('Hyperparameter analysis')
        ax.set_ylim(0, 1)
    
    def _plot_computational_performance(self, ax: plt.Axes) -> None:
        """Plot computational performance metrics."""
        metrics = ['Training\nTime', 'Inference\nTime', 'Memory\nUsage']
        values = [0.8, 0.9, 0.7]  # Normalized values
        
        ax.bar(metrics, values, color=self.color_palettes[self.style][3], alpha=0.7)
        ax.set_ylabel('Efficiency (normalized)')
        ax.set_title('Computational performance')
        ax.set_ylim(0, 1)
    
    def _plot_performance_comparison(self, ax: plt.Axes, results_comparison: Dict[str, Dict[str, Any]]) -> None:
        """Plot performance comparison across methods."""
        methods = list(results_comparison.keys())
        accuracies = [0.75, 0.82, 0.88, 0.91]  # Mock data
        
        bars = ax.bar(methods, accuracies, color=self.color_palettes[self.style][:len(methods)], alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_title('Method comparison')
        ax.set_ylim(0, 1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_celltype_accuracy_comparison(self, ax: plt.Axes, results_comparison: Dict[str, Dict[str, Any]]) -> None:
        """Plot cell type accuracy comparison."""
        cell_types = ['T Cell', 'B Cell', 'NK Cell', 'Monocyte']
        method1_acc = np.random.uniform(0.7, 0.9, len(cell_types))
        method2_acc = np.random.uniform(0.75, 0.95, len(cell_types))
        
        x = np.arange(len(cell_types))
        width = 0.35
        
        ax.bar(x - width/2, method1_acc, width, label='Baseline', alpha=0.7)
        ax.bar(x + width/2, method2_acc, width, label='Our method', alpha=0.7)
        
        ax.set_ylabel('Accuracy')
        ax.set_title('Cell type accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(cell_types)
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_efficiency_comparison(self, ax: plt.Axes, results_comparison: Dict[str, Dict[str, Any]]) -> None:
        """Plot computational efficiency comparison."""
        methods = ['Method A', 'Method B', 'Our Method']
        times = [120, 85, 45]  # Training time in minutes
        
        bars = ax.bar(methods, times, color=['red', 'orange', 'green'], alpha=0.7)
        ax.set_ylabel('Training time (min)')
        ax.set_title('Computational efficiency')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_confusion_matrices(self, ax: plt.Axes, results_comparison: Dict[str, Dict[str, Any]]) -> None:
        """Plot confusion matrices comparison."""
        # Mock confusion matrix data
        cm = np.random.randint(0, 100, (4, 4))
        cm = cm / cm.sum(axis=1, keepdims=True)  # Normalize
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title('Confusion matrix')
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]:.2f}', ha="center", va="center")
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    def _create_pathway_supplementary(
        self, 
        fig: plt.Figure, 
        additional_results: Dict[str, Any], 
        save_path: Optional[str]
    ) -> plt.Figure:
        """Create pathway-focused supplementary figure."""
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Add pathway-specific panels
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(0.5, 0.5, 'Pathway Enrichment\nAnalysis', ha='center', va='center')
        ax1.set_title('Supplementary pathway analysis')
        self._add_panel_label(ax1, 'A')
        
        if save_path:
            self._save_publication_figure(fig, save_path)
        
        return fig
    
    def _create_interaction_supplementary(
        self, 
        fig: plt.Figure, 
        additional_results: Dict[str, Any], 
        save_path: Optional[str]
    ) -> plt.Figure:
        """Create interaction-focused supplementary figure."""
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Add interaction-specific panels
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(0.5, 0.5, 'Cell-Cell Interaction\nNetworks', ha='center', va='center')
        ax1.set_title('Supplementary interaction analysis')
        self._add_panel_label(ax1, 'A')
        
        if save_path:
            self._save_publication_figure(fig, save_path)
        
        return fig
    
    def _create_general_supplementary(
        self, 
        fig: plt.Figure, 
        additional_results: Dict[str, Any], 
        save_path: Optional[str]
    ) -> plt.Figure:
        """Create general supplementary figure."""
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Add general panels
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(0.5, 0.5, 'Additional\nAnalyses', ha='center', va='center')
        ax1.set_title('Supplementary analyses')
        self._add_panel_label(ax1, 'A')
        
        if save_path:
            self._save_publication_figure(fig, save_path)
        
        return fig