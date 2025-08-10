"""
Spatial plotting functionality for visualizing spatial transcriptomics data.

This module provides comprehensive plotting capabilities for spatial data
including cell type maps, gene expression, and prediction overlays.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Union, Tuple
from pathlib import Path
import warnings

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from ..data.base import BaseSpatialDataset


class SpatialPlotter:
    """
    Comprehensive spatial plotting engine for spatial transcriptomics data.
    
    Provides methods for visualizing spatial data with various overlays
    including cell types, gene expression, predictions, and interactions.
    """
    
    def __init__(
        self,
        style: str = "default",
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 300
    ):
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        
        # Set style
        if style == "publication":
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
        elif style == "dark":
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
    
    def plot_spatial(
        self,
        dataset: BaseSpatialDataset,
        color_by: str = "cell_type",
        predictions: Optional[Dict[str, Any]] = None,
        gene_name: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_legend: bool = True,
        point_size: float = 20,
        alpha: float = 0.8,
        cmap: str = "viridis"
    ) -> plt.Figure:
        """
        Create spatial plot of the dataset.
        
        Args:
            dataset: Spatial dataset to plot
            color_by: What to color points by ('cell_type', 'gene', 'prediction', 'confidence')
            predictions: Prediction results from model
            gene_name: Gene name for expression plotting
            title: Plot title
            save_path: Path to save the plot
            show_legend: Whether to show legend
            point_size: Size of points
            alpha: Point transparency
            cmap: Colormap for continuous values
            
        Returns:
            Matplotlib figure
        """
        # Get spatial coordinates
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Determine colors
        colors, color_labels = self._get_colors(
            dataset, color_by, predictions, gene_name
        )
        
        # Create scatter plot
        if isinstance(colors, np.ndarray) and colors.dtype in [np.float32, np.float64]:
            # Continuous coloring
            scatter = ax.scatter(
                spatial_coords[:, 0],
                spatial_coords[:, 1],
                c=colors,
                s=point_size,
                alpha=alpha,
                cmap=cmap,
                edgecolors='none'
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label(color_by.replace('_', ' ').title())
            
        else:
            # Categorical coloring
            unique_labels = np.unique(color_labels)
            palette = sns.color_palette("husl", len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = color_labels == label
                ax.scatter(
                    spatial_coords[mask, 0],
                    spatial_coords[mask, 1],
                    c=[palette[i]],
                    s=point_size,
                    alpha=alpha,
                    label=label,
                    edgecolors='none'
                )
            
            # Add legend
            if show_legend and len(unique_labels) <= 20:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Formatting
        ax.set_xlabel('X Position (μm)')
        ax.set_ylabel('Y Position (μm)')
        ax.set_aspect('equal')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Spatial Plot - {color_by.replace("_", " ").title()}')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_gene_expression(
        self,
        dataset: BaseSpatialDataset,
        gene_names: Union[str, List[str]],
        ncols: int = 2,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        point_size: float = 15,
        cmap: str = "viridis"
    ) -> plt.Figure:
        """
        Plot spatial gene expression patterns.
        
        Args:
            dataset: Spatial dataset
            gene_names: Gene name(s) to plot
            ncols: Number of columns for subplots
            title: Overall title
            save_path: Path to save plot
            point_size: Size of points
            cmap: Colormap
            
        Returns:
            Matplotlib figure
        """
        if isinstance(gene_names, str):
            gene_names = [gene_names]
        
        # Calculate subplot layout
        nrows = (len(gene_names) + ncols - 1) // ncols
        
        # Create figure
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(self.figsize[0] * ncols, self.figsize[1] * nrows),
            dpi=self.dpi
        )
        
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Get spatial coordinates
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        gene_expression = data.x.numpy()
        
        # Plot each gene
        for i, gene_name in enumerate(gene_names):
            ax = axes[i]
            
            # Get gene expression
            if gene_name in dataset.gene_names:
                gene_idx = dataset.gene_names.index(gene_name)
                expression = gene_expression[:, gene_idx]
            else:
                warnings.warn(f"Gene {gene_name} not found in dataset")
                expression = np.zeros(len(spatial_coords))
            
            # Create scatter plot
            scatter = ax.scatter(
                spatial_coords[:, 0],
                spatial_coords[:, 1],
                c=expression,
                s=point_size,
                cmap=cmap,
                edgecolors='none'
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Expression')
            
            # Formatting
            ax.set_xlabel('X Position (μm)')
            ax.set_ylabel('Y Position (μm)')
            ax.set_title(gene_name)
            ax.set_aspect('equal')
            
            # Remove spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Hide unused subplots
        for i in range(len(gene_names), len(axes)):
            axes[i].set_visible(False)
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_predictions_overlay(
        self,
        dataset: BaseSpatialDataset,
        predictions: Dict[str, Any],
        overlay_type: str = "confidence",
        threshold: float = 0.5,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        point_size: float = 20
    ) -> plt.Figure:
        """
        Plot predictions with confidence overlay.
        
        Args:
            dataset: Spatial dataset
            predictions: Model predictions
            overlay_type: Type of overlay ('confidence', 'uncertainty', 'predictions')
            threshold: Threshold for filtering
            title: Plot title
            save_path: Path to save
            point_size: Point size
            
        Returns:
            Matplotlib figure
        """
        # Get spatial coordinates
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 2, self.figsize[1]))
        
        # Left plot: All predictions
        ax1 = axes[0]
        
        if 'predictions' in predictions:
            pred_labels = predictions.get('cell_type_names', 
                                        [f'Type_{i}' for i in range(max(predictions['predictions']) + 1)])
            
            colors = predictions['predictions']
            unique_preds = np.unique(colors)
            palette = sns.color_palette("husl", len(unique_preds))
            
            for i, pred in enumerate(unique_preds):
                mask = colors == pred
                ax1.scatter(
                    spatial_coords[mask, 0],
                    spatial_coords[mask, 1],
                    c=[palette[i]],
                    s=point_size,
                    alpha=0.8,
                    label=pred_labels[pred] if pred < len(pred_labels) else f'Type_{pred}',
                    edgecolors='none'
                )
            
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax1.set_xlabel('X Position (μm)')
        ax1.set_ylabel('Y Position (μm)')
        ax1.set_title('All Predictions')
        ax1.set_aspect('equal')
        
        # Right plot: High confidence only
        ax2 = axes[1]
        
        if overlay_type == "confidence" and 'confidence' in predictions:
            confidence = predictions['confidence']
            high_conf_mask = confidence >= threshold
            
            # Plot low confidence in gray
            low_conf_mask = ~high_conf_mask
            ax2.scatter(
                spatial_coords[low_conf_mask, 0],
                spatial_coords[low_conf_mask, 1],
                c='lightgray',
                s=point_size * 0.5,
                alpha=0.3,
                edgecolors='none'
            )
            
            # Plot high confidence with colors
            if np.any(high_conf_mask):
                high_conf_colors = colors[high_conf_mask]
                
                for i, pred in enumerate(unique_preds):
                    pred_mask = high_conf_colors == pred
                    if np.any(pred_mask):
                        high_conf_indices = np.where(high_conf_mask)[0]
                        plot_indices = high_conf_indices[pred_mask]
                        
                        ax2.scatter(
                            spatial_coords[plot_indices, 0],
                            spatial_coords[plot_indices, 1],
                            c=[palette[i]],
                            s=point_size,
                            alpha=0.8,
                            label=pred_labels[pred] if pred < len(pred_labels) else f'Type_{pred}',
                            edgecolors='none'
                        )
            
            ax2.set_title(f'High Confidence (≥{threshold})')\n        
        elif overlay_type == "uncertainty" and 'uncertainty' in predictions:
            scatter = ax2.scatter(
                spatial_coords[:, 0],
                spatial_coords[:, 1],
                c=predictions['uncertainty'],
                s=point_size,
                cmap='viridis_r',  # Reverse so low uncertainty is bright
                edgecolors='none'
            )
            
            cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
            cbar.set_label('Uncertainty')
            ax2.set_title('Prediction Uncertainty')
        
        ax2.set_xlabel('X Position (μm)')
        ax2.set_ylabel('Y Position (μm)')
        ax2.set_aspect('equal')
        
        # Remove spines
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_interactions(
        self,
        dataset: BaseSpatialDataset,
        interactions: Dict[str, Any],
        show_top_n: int = 50,
        edge_width_scale: float = 2.0,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        point_size: float = 15
    ) -> plt.Figure:
        """
        Plot cell-cell interactions on spatial map.
        
        Args:
            dataset: Spatial dataset
            interactions: Interaction predictions
            show_top_n: Number of top interactions to show
            edge_width_scale: Scale factor for edge widths
            title: Plot title
            save_path: Path to save
            point_size: Point size
            
        Returns:
            Matplotlib figure
        """
        # Get spatial coordinates
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot all cells
        ax.scatter(
            spatial_coords[:, 0],
            spatial_coords[:, 1],
            c='lightgray',
            s=point_size,
            alpha=0.6,
            edgecolors='none'
        )
        
        # Plot interactions
        if 'interactions' in interactions:
            interaction_list = interactions['interactions'][:show_top_n]
            
            for interaction in interaction_list:
                cell_i = interaction['cell_i']
                cell_j = interaction['cell_j']
                confidence = interaction['confidence']
                
                # Draw edge
                ax.plot(
                    [spatial_coords[cell_i, 0], spatial_coords[cell_j, 0]],
                    [spatial_coords[cell_i, 1], spatial_coords[cell_j, 1]],
                    color='red',
                    alpha=confidence,
                    linewidth=confidence * edge_width_scale,
                    zorder=1
                )
            
            # Highlight interacting cells
            interacting_cells = set()
            for interaction in interaction_list:
                interacting_cells.add(interaction['cell_i'])
                interacting_cells.add(interaction['cell_j'])
            
            interacting_indices = list(interacting_cells)
            ax.scatter(
                spatial_coords[interacting_indices, 0],
                spatial_coords[interacting_indices, 1],
                c='red',
                s=point_size * 1.5,
                alpha=0.8,
                edgecolors='darkred',
                linewidth=0.5,
                zorder=2
            )
        
        # Formatting
        ax.set_xlabel('X Position (μm)')
        ax.set_ylabel('Y Position (μm)')
        ax.set_aspect('equal')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Cell-Cell Interactions (Top {show_top_n})')
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _get_colors(
        self,
        dataset: BaseSpatialDataset,
        color_by: str,
        predictions: Optional[Dict[str, Any]] = None,
        gene_name: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get colors and labels for plotting.
        
        Args:
            dataset: Spatial dataset
            color_by: What to color by
            predictions: Model predictions
            gene_name: Gene name for expression
            
        Returns:
            Tuple of (colors, labels)
        """
        data = dataset.get(0)
        
        if color_by == "cell_type" and hasattr(data, 'cell_type'):
            return data.cell_type, data.cell_type
            
        elif color_by == "gene" and gene_name:
            if gene_name in dataset.gene_names:
                gene_idx = dataset.gene_names.index(gene_name)
                expression = data.x[:, gene_idx].numpy()
                return expression, expression
            else:
                warnings.warn(f"Gene {gene_name} not found")
                return np.zeros(data.x.size(0)), np.zeros(data.x.size(0))
                
        elif color_by == "prediction" and predictions and 'predictions' in predictions:
            pred_array = predictions['predictions']
            if isinstance(pred_array, list):
                pred_array = np.array(pred_array)
            return pred_array, pred_array
            
        elif color_by == "confidence" and predictions and 'confidence' in predictions:
            conf_array = predictions['confidence']
            if isinstance(conf_array, list):
                conf_array = np.array(conf_array)
            return conf_array, conf_array
            
        else:
            # Default: color all points the same
            num_points = data.x.size(0)
            return np.zeros(num_points), np.array(['Cell'] * num_points)
    
    def create_summary_plot(
        self,
        dataset: BaseSpatialDataset,
        predictions: Optional[Dict[str, Any]] = None,
        interactions: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive summary plot with multiple panels.
        
        Args:
            dataset: Spatial dataset
            predictions: Model predictions
            interactions: Interaction predictions
            save_path: Path to save
            
        Returns:
            Matplotlib figure with multiple panels
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12), dpi=self.dpi)
        
        # Panel 1: Spatial cell types
        ax1 = plt.subplot(2, 3, 1)
        self._plot_panel(ax1, dataset, "cell_type", "Cell Types")
        
        # Panel 2: Predictions
        ax2 = plt.subplot(2, 3, 2)
        if predictions:
            self._plot_panel(ax2, dataset, "prediction", "Predictions", predictions)
        else:
            ax2.text(0.5, 0.5, 'No predictions available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Predictions')
        
        # Panel 3: Confidence
        ax3 = plt.subplot(2, 3, 3)
        if predictions and 'confidence' in predictions:
            self._plot_panel(ax3, dataset, "confidence", "Prediction Confidence", predictions)
        else:
            ax3.text(0.5, 0.5, 'No confidence scores available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Confidence')
        
        # Panel 4: Interactions
        ax4 = plt.subplot(2, 3, (4, 5))  # Span two columns
        if interactions:
            self._plot_interactions_panel(ax4, dataset, interactions)
        else:
            ax4.text(0.5, 0.5, 'No interactions available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Cell-Cell Interactions')
        
        # Panel 5: Statistics
        ax5 = plt.subplot(2, 3, 6)
        self._plot_statistics_panel(ax5, dataset, predictions, interactions)
        
        plt.suptitle('Spatial-Omics GFM Analysis Summary', fontsize=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _plot_panel(
        self,
        ax: plt.Axes,
        dataset: BaseSpatialDataset,
        color_by: str,
        title: str,
        predictions: Optional[Dict[str, Any]] = None
    ) -> None:
        """Plot a single panel."""
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        colors, labels = self._get_colors(dataset, color_by, predictions)
        
        if isinstance(colors, np.ndarray) and colors.dtype in [np.float32, np.float64]:
            scatter = ax.scatter(
                spatial_coords[:, 0], spatial_coords[:, 1],
                c=colors, s=15, alpha=0.8, cmap='viridis'
            )
            plt.colorbar(scatter, ax=ax, shrink=0.8)
        else:
            unique_labels = np.unique(labels)
            palette = sns.color_palette("husl", len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    spatial_coords[mask, 0], spatial_coords[mask, 1],
                    c=[palette[i]], s=15, alpha=0.8, label=str(label)
                )
        
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def _plot_interactions_panel(
        self,
        ax: plt.Axes,
        dataset: BaseSpatialDataset,
        interactions: Dict[str, Any]
    ) -> None:
        """Plot interactions panel."""
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        # Plot cells
        ax.scatter(
            spatial_coords[:, 0], spatial_coords[:, 1],
            c='lightgray', s=10, alpha=0.6
        )
        
        # Plot top interactions
        if 'interactions' in interactions:
            top_interactions = interactions['interactions'][:20]
            
            for interaction in top_interactions:
                cell_i = interaction['cell_i']
                cell_j = interaction['cell_j']
                confidence = interaction['confidence']
                
                ax.plot(
                    [spatial_coords[cell_i, 0], spatial_coords[cell_j, 0]],
                    [spatial_coords[cell_i, 1], spatial_coords[cell_j, 1]],
                    color='red', alpha=confidence, linewidth=confidence * 2
                )
        
        ax.set_title('Cell-Cell Interactions (Top 20)')
        ax.set_aspect('equal')
    
    def _plot_statistics_panel(
        self,
        ax: plt.Axes,
        dataset: BaseSpatialDataset,
        predictions: Optional[Dict[str, Any]] = None,
        interactions: Optional[Dict[str, Any]] = None
    ) -> None:
        """Plot statistics panel."""
        ax.axis('off')
        
        stats_text = f"Dataset Statistics:\n"
        stats_text += f"• Cells: {dataset.num_cells:,}\n"
        stats_text += f"• Genes: {dataset.num_genes:,}\n\n"
        
        if predictions:
            stats_text += f"Predictions:\n"
            if 'total_cells' in predictions:
                stats_text += f"• Total cells: {predictions['total_cells']:,}\n"
            if 'num_high_confidence' in predictions:
                stats_text += f"• High confidence: {predictions['num_high_confidence']:,}\n"
            stats_text += "\n"
        
        if interactions:
            stats_text += f"Interactions:\n"
            if 'num_interactions' in interactions:
                stats_text += f"• Total: {interactions['num_interactions']:,}\n"
            if 'distance_threshold' in interactions:
                stats_text += f"• Distance threshold: {interactions['distance_threshold']:.0f}μm\n"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        ax.set_title('Analysis Statistics')
    
    def create_interactive_spatial_plot(
        self,
        dataset: BaseSpatialDataset,
        color_by: str = "cell_type",
        predictions: Optional[Dict[str, Any]] = None,
        gene_name: Optional[str] = None,
        point_size: float = 5,
        opacity: float = 0.8,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive spatial plot using Plotly.
        
        Args:
            dataset: Spatial dataset to plot
            color_by: What to color points by
            predictions: Model predictions
            gene_name: Gene name for expression coloring
            point_size: Size of points
            opacity: Point opacity
            title: Plot title
            
        Returns:
            Interactive Plotly figure
        """
        # Get spatial coordinates
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        # Prepare data for plotting
        df_data = {
            'x': spatial_coords[:, 0],
            'y': spatial_coords[:, 1],
            'cell_id': list(range(len(spatial_coords)))
        }
        
        # Determine colors
        colors, labels = self._get_colors(dataset, color_by, predictions, gene_name)
        
        if isinstance(colors, np.ndarray) and colors.dtype in [np.float32, np.float64]:
            # Continuous coloring
            df_data['value'] = colors
            df = pd.DataFrame(df_data)
            
            fig = px.scatter(
                df, x='x', y='y', color='value',
                color_continuous_scale='viridis',
                hover_data={'cell_id': True},
                title=title or f'Interactive Spatial Plot - {color_by.replace("_", " ").title()}'
            )
        else:
            # Categorical coloring
            df_data['category'] = labels
            df = pd.DataFrame(df_data)
            
            fig = px.scatter(
                df, x='x', y='y', color='category',
                hover_data={'cell_id': True},
                title=title or f'Interactive Spatial Plot - {color_by.replace("_", " ").title()}'
            )
        
        # Update layout
        fig.update_traces(marker=dict(size=point_size, opacity=opacity))
        fig.update_layout(
            xaxis_title='X Position (μm)',
            yaxis_title='Y Position (μm)',
            template='plotly_white',
            width=800,
            height=600
        )
        
        # Ensure equal aspect ratio
        fig.update_yaxis(scaleanchor="x", scaleratio=1)
        
        return fig
    
    def create_plotly_gene_expression_plot(
        self,
        dataset: BaseSpatialDataset,
        gene_names: Union[str, List[str]],
        ncols: int = 2,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive multi-gene expression plot using Plotly.
        
        Args:
            dataset: Spatial dataset
            gene_names: Gene name(s) to plot
            ncols: Number of columns for subplots
            title: Overall title
            
        Returns:
            Interactive Plotly figure with subplots
        """
        if isinstance(gene_names, str):
            gene_names = [gene_names]
        
        # Calculate subplot layout
        nrows = (len(gene_names) + ncols - 1) // ncols
        
        # Create subplots
        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=gene_names,
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )
        
        # Get spatial coordinates
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        gene_expression = data.x.numpy()
        
        # Plot each gene
        for i, gene_name in enumerate(gene_names):
            row = i // ncols + 1
            col = i % ncols + 1
            
            # Get gene expression
            if gene_name in dataset.gene_names:
                gene_idx = dataset.gene_names.index(gene_name)
                expression = gene_expression[:, gene_idx]
            else:
                warnings.warn(f"Gene {gene_name} not found in dataset")
                expression = np.zeros(len(spatial_coords))
            
            # Create scatter trace
            scatter = go.Scatter(
                x=spatial_coords[:, 0],
                y=spatial_coords[:, 1],
                mode='markers',
                marker=dict(
                    color=expression,
                    colorscale='viridis',
                    size=4,
                    opacity=0.8,
                    showscale=True if i == 0 else False,
                    colorbar=dict(title="Expression") if i == 0 else None
                ),
                hovertemplate=f"{gene_name}<br>X: %{{x}}<br>Y: %{{y}}<br>Expression: %{{marker.color}}<extra></extra>",
                name=gene_name,
                showlegend=False
            )
            
            fig.add_trace(scatter, row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title=title or "Interactive Gene Expression Patterns",
            template='plotly_white',
            width=1000,
            height=600
        )
        
        # Update axes
        for i in range(1, nrows + 1):
            for j in range(1, ncols + 1):
                fig.update_xaxes(title_text="X Position (μm)", row=i, col=j)
                fig.update_yaxes(title_text="Y Position (μm)", row=i, col=j)
        
        return fig
    
    def plot_spatial_regions(
        self,
        dataset: BaseSpatialDataset,
        regions: Dict[str, Any],
        region_colors: Optional[Dict[str, str]] = None,
        show_boundaries: bool = True,
        boundary_alpha: float = 0.7,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot spatial regions/domains identified in the tissue.
        
        Args:
            dataset: Spatial dataset
            regions: Dictionary containing region assignments
            region_colors: Custom colors for regions
            show_boundaries: Whether to show region boundaries
            boundary_alpha: Alpha for boundary lines
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Get spatial coordinates
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Get region assignments
        if 'region_assignments' in regions:
            region_labels = regions['region_assignments']
        else:
            # Generate mock regions for demonstration
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42)
            region_labels = kmeans.fit_predict(spatial_coords)
        
        # Set up colors
        unique_regions = np.unique(region_labels)
        if region_colors is None:
            colors = sns.color_palette("Set2", len(unique_regions))
            region_colors = {region: colors[i] for i, region in enumerate(unique_regions)}
        
        # Plot regions
        for region in unique_regions:
            mask = region_labels == region
            ax.scatter(
                spatial_coords[mask, 0],
                spatial_coords[mask, 1],
                c=[region_colors.get(region, 'gray')],
                s=20,
                alpha=0.8,
                label=f'Region {region}',
                edgecolors='none'
            )
        
        # Add boundaries if requested
        if show_boundaries:
            self._add_region_boundaries(ax, spatial_coords, region_labels, boundary_alpha)
        
        # Formatting
        ax.set_xlabel('X Position (μm)', fontsize=12)
        ax.set_ylabel('Y Position (μm)', fontsize=12)
        ax.set_aspect('equal')
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Spatial Regions/Domains', fontsize=14)
        
        if len(unique_regions) <= 10:  # Only show legend if not too many regions
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_spatial_neighborhoods(
        self,
        dataset: BaseSpatialDataset,
        neighborhoods: Dict[str, Any],
        radius: float = 50.0,
        show_connections: bool = True,
        max_connections: int = 20,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot spatial neighborhoods and cell-cell proximities.
        
        Args:
            dataset: Spatial dataset
            neighborhoods: Dictionary containing neighborhood information
            radius: Neighborhood radius in micrometers
            show_connections: Whether to show cell connections
            max_connections: Maximum connections to display per cell
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Get spatial coordinates
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot all cells
        ax.scatter(
            spatial_coords[:, 0],
            spatial_coords[:, 1],
            c='lightblue',
            s=15,
            alpha=0.6,
            edgecolors='none',
            zorder=1
        )
        
        # Add neighborhood connections
        if show_connections:
            # Calculate distances
            distances = cdist(spatial_coords, spatial_coords)
            
            # Find neighbors within radius
            neighbor_count = 0
            for i in range(len(spatial_coords)):
                if neighbor_count >= max_connections:
                    break
                
                neighbors = np.where((distances[i, :] <= radius) & (distances[i, :] > 0))[0]
                
                for j in neighbors[:5]:  # Limit connections per cell
                    if neighbor_count >= max_connections:
                        break
                    
                    ax.plot(
                        [spatial_coords[i, 0], spatial_coords[j, 0]],
                        [spatial_coords[i, 1], spatial_coords[j, 1]],
                        color='red',
                        alpha=0.3,
                        linewidth=0.5,
                        zorder=2
                    )
                    neighbor_count += 1
        
        # Highlight central cells in neighborhoods
        if 'neighborhood_centers' in neighborhoods:
            centers = neighborhoods['neighborhood_centers']
            ax.scatter(
                spatial_coords[centers, 0],
                spatial_coords[centers, 1],
                c='red',
                s=50,
                alpha=0.8,
                marker='*',
                edgecolors='darkred',
                linewidth=1,
                label='Neighborhood Centers',
                zorder=3
            )
        
        # Add radius circles for visualization
        if 'neighborhood_centers' in neighborhoods:
            centers = neighborhoods['neighborhood_centers'][:10]  # Limit for clarity
            for center_idx in centers:
                center = spatial_coords[center_idx]
                circle = plt.Circle(center, radius, fill=False, 
                                  color='red', alpha=0.5, linewidth=1)
                ax.add_patch(circle)
        
        # Formatting
        ax.set_xlabel('X Position (μm)', fontsize=12)
        ax.set_ylabel('Y Position (μm)', fontsize=12)
        ax.set_aspect('equal')
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'Spatial Neighborhoods (radius: {radius}μm)', fontsize=14)
        
        if 'neighborhood_centers' in neighborhoods:
            ax.legend()
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_tissue_morphology(
        self,
        dataset: BaseSpatialDataset,
        morphology_features: Optional[Dict[str, Any]] = None,
        feature_name: str = "density",
        overlay_predictions: bool = False,
        predictions: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot tissue morphological features.
        
        Args:
            dataset: Spatial dataset
            morphology_features: Dictionary with morphological features
            feature_name: Name of feature to visualize
            overlay_predictions: Whether to overlay predictions
            predictions: Model predictions for overlay
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Get spatial coordinates
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate or extract morphological features
        if morphology_features and feature_name in morphology_features:
            feature_values = morphology_features[feature_name]
        else:
            # Calculate cell density as default morphological feature
            from scipy.spatial.distance import pdist, squareform
            
            # Calculate local density (number of neighbors within 100μm)
            distances = squareform(pdist(spatial_coords))
            feature_values = np.sum(distances <= 100, axis=1) - 1  # Exclude self
        
        # Create the base plot
        scatter = ax.scatter(
            spatial_coords[:, 0],
            spatial_coords[:, 1],
            c=feature_values,
            s=25,
            alpha=0.8,
            cmap='viridis',
            edgecolors='none'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(feature_name.replace('_', ' ').title())
        
        # Overlay predictions if requested
        if overlay_predictions and predictions and 'predictions' in predictions:
            # Show predictions as contour or boundary overlay
            pred_labels = predictions['predictions']
            unique_preds = np.unique(pred_labels)
            
            # Create prediction boundaries
            for pred in unique_preds:
                mask = pred_labels == pred
                if np.sum(mask) > 5:  # Only if enough points
                    from scipy.spatial import ConvexHull
                    try:
                        hull = ConvexHull(spatial_coords[mask])
                        for simplex in hull.simplices:
                            ax.plot(spatial_coords[mask][simplex, 0], 
                                   spatial_coords[mask][simplex, 1], 
                                   'r-', alpha=0.3, linewidth=1)
                    except:
                        pass  # Skip if hull computation fails
        
        # Formatting
        ax.set_xlabel('X Position (μm)', fontsize=12)
        ax.set_ylabel('Y Position (μm)', fontsize=12)
        ax.set_aspect('equal')
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            title_text = f'Tissue Morphology - {feature_name.replace("_", " ").title()}'
            if overlay_predictions:
                title_text += ' with Predictions'
            ax.set_title(title_text, fontsize=14)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _add_region_boundaries(
        self,
        ax: plt.Axes,
        coordinates: np.ndarray,
        region_labels: np.ndarray,
        alpha: float = 0.7
    ) -> None:
        """Add region boundaries to the plot."""
        from scipy.spatial import ConvexHull
        
        unique_regions = np.unique(region_labels)
        
        for region in unique_regions:
            mask = region_labels == region
            region_coords = coordinates[mask]
            
            if len(region_coords) >= 3:  # Need at least 3 points for hull
                try:
                    hull = ConvexHull(region_coords)
                    # Plot the convex hull boundary
                    for simplex in hull.simplices:
                        ax.plot(region_coords[simplex, 0], 
                               region_coords[simplex, 1],
                               'k-', alpha=alpha, linewidth=1)
                except:
                    # Skip if convex hull computation fails
                    pass