"""
Cell-cell interaction network visualization for spatial transcriptomics data.

This module provides comprehensive visualization capabilities for cell-cell
interactions including network graphs, spatial overlays, and interaction heatmaps.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Optional, Dict, List, Any, Union, Tuple
from pathlib import Path
import warnings
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

from ..data.base import BaseSpatialDataset


class InteractionNetworkPlotter:
    """
    Comprehensive interaction network visualization engine.
    
    Provides methods for visualizing cell-cell interactions as network graphs,
    spatial overlays, and statistical summaries with both static and interactive plots.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 10),
        dpi: int = 300,
        style: str = "default"
    ):
        """
        Initialize the interaction network plotter.
        
        Args:
            figsize: Default figure size for matplotlib plots
            dpi: Resolution for saved figures
            style: Plot style ('default', 'publication', 'dark')
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        # Set matplotlib style
        if style == "publication":
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
        elif style == "dark":
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
    
    def plot_interaction_network(
        self,
        interactions: Dict[str, Any],
        dataset: Optional[BaseSpatialDataset] = None,
        cell_types: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        max_interactions: int = 100,
        layout: str = "spring",
        node_size_scale: float = 300,
        edge_width_scale: float = 5.0,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot cell-cell interactions as a network graph.
        
        Args:
            interactions: Interaction predictions dictionary
            dataset: Optional spatial dataset for additional context
            cell_types: List of cell type names
            confidence_threshold: Minimum confidence for displayed interactions
            max_interactions: Maximum number of interactions to display
            layout: Network layout algorithm ('spring', 'circular', 'kamada_kawai')
            node_size_scale: Scale factor for node sizes
            edge_width_scale: Scale factor for edge widths
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Create network graph
        G = nx.Graph()
        
        # Extract interactions
        if 'interactions' not in interactions:
            raise ValueError("Interactions dictionary must contain 'interactions' key")
        
        interaction_list = interactions['interactions']
        
        # Filter by confidence and limit number
        filtered_interactions = [
            inter for inter in interaction_list
            if inter.get('confidence', 0) >= confidence_threshold
        ][:max_interactions]
        
        # Add nodes and edges
        nodes_set = set()
        for interaction in filtered_interactions:
            cell_i = interaction['cell_i']
            cell_j = interaction['cell_j']
            confidence = interaction['confidence']
            
            # Add nodes
            nodes_set.add(cell_i)
            nodes_set.add(cell_j)
            
            # Add edge with weight
            G.add_edge(cell_i, cell_j, weight=confidence)
        
        # Add isolated nodes if they have cell types
        if cell_types:
            for i, cell_type in enumerate(cell_types):
                if i not in nodes_set:
                    G.add_node(i)
        
        # Calculate layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Prepare node colors and sizes
        if cell_types and dataset:
            # Color by cell type
            unique_cell_types = list(set(cell_types))
            color_map = dict(zip(unique_cell_types, sns.color_palette("husl", len(unique_cell_types))))
            node_colors = [color_map.get(cell_types[node], 'gray') for node in G.nodes()]
        else:
            node_colors = 'lightblue'
        
        # Node sizes based on degree
        node_degrees = dict(G.degree())
        node_sizes = [node_degrees.get(node, 1) * node_size_scale for node in G.nodes()]
        
        # Edge widths based on confidence
        edge_weights = [G[u][v]['weight'] * edge_width_scale for u, v in G.edges()]
        
        # Draw network
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            linewidths=0.5,
            edgecolors='black'
        )
        
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=edge_weights,
            alpha=0.6,
            edge_color='red'
        )
        
        # Add labels for high-degree nodes
        high_degree_nodes = {node: degree for node, degree in node_degrees.items() if degree > 3}
        if high_degree_nodes:
            labels = {node: str(node) for node in high_degree_nodes.keys()}
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
        
        # Formatting
        ax.set_aspect('equal')
        ax.axis('off')
        
        if title:
            ax.set_title(title, fontsize=16, pad=20)
        else:
            ax.set_title(f'Cell-Cell Interaction Network (n={len(filtered_interactions)})', 
                        fontsize=16, pad=20)
        
        # Add legend if cell types are available
        if cell_types and dataset:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color_map[ct], markersize=10, label=ct)
                for ct in unique_cell_types[:10]  # Limit legend entries
            ]
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_spatial_interactions(
        self,
        dataset: BaseSpatialDataset,
        interactions: Dict[str, Any],
        confidence_threshold: float = 0.5,
        max_interactions: int = 100,
        show_all_cells: bool = True,
        interaction_color: str = 'red',
        cell_color: str = 'lightgray',
        point_size: float = 20,
        edge_alpha: float = 0.6,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot interactions overlaid on spatial coordinates.
        
        Args:
            dataset: Spatial dataset
            interactions: Interaction predictions
            confidence_threshold: Minimum confidence for displayed interactions
            max_interactions: Maximum number of interactions to show
            show_all_cells: Whether to show all cells or only interacting ones
            interaction_color: Color for interaction edges
            cell_color: Color for non-interacting cells
            point_size: Size of cell points
            edge_alpha: Transparency of interaction edges
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
        
        # Plot all cells if requested
        if show_all_cells:
            ax.scatter(
                spatial_coords[:, 0],
                spatial_coords[:, 1],
                c=cell_color,
                s=point_size,
                alpha=0.4,
                edgecolors='none',
                zorder=1
            )
        
        # Filter interactions by confidence
        if 'interactions' not in interactions:
            raise ValueError("Interactions dictionary must contain 'interactions' key")
        
        interaction_list = interactions['interactions']
        filtered_interactions = [
            inter for inter in interaction_list
            if inter.get('confidence', 0) >= confidence_threshold
        ][:max_interactions]
        
        # Plot interactions
        interacting_cells = set()
        for interaction in filtered_interactions:
            cell_i = interaction['cell_i']
            cell_j = interaction['cell_j']
            confidence = interaction['confidence']
            
            # Skip if indices are out of bounds
            if cell_i >= len(spatial_coords) or cell_j >= len(spatial_coords):
                continue
            
            interacting_cells.add(cell_i)
            interacting_cells.add(cell_j)
            
            # Draw interaction edge
            ax.plot(
                [spatial_coords[cell_i, 0], spatial_coords[cell_j, 0]],
                [spatial_coords[cell_i, 1], spatial_coords[cell_j, 1]],
                color=interaction_color,
                alpha=min(edge_alpha * confidence, 1.0),
                linewidth=confidence * 3,
                zorder=2
            )
        
        # Highlight interacting cells
        if interacting_cells:
            interacting_indices = list(interacting_cells)
            ax.scatter(
                spatial_coords[interacting_indices, 0],
                spatial_coords[interacting_indices, 1],
                c=interaction_color,
                s=point_size * 1.2,
                alpha=0.8,
                edgecolors='darkred',
                linewidth=0.5,
                zorder=3
            )
        
        # Formatting
        ax.set_xlabel('X Position (μm)', fontsize=12)
        ax.set_ylabel('Y Position (μm)', fontsize=12)
        ax.set_aspect('equal')
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'Spatial Cell-Cell Interactions (n={len(filtered_interactions)})', 
                        fontsize=14)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_interaction_heatmap(
        self,
        interactions: Dict[str, Any],
        cell_types: Optional[List[str]] = None,
        aggregation: str = "count",
        min_interactions: int = 1,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create heatmap of cell type interactions.
        
        Args:
            interactions: Interaction predictions
            cell_types: List of cell type names for each cell
            aggregation: How to aggregate interactions ('count', 'mean_confidence', 'sum_confidence')
            min_interactions: Minimum interactions to include cell type
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not cell_types:
            raise ValueError("Cell types must be provided for interaction heatmap")
        
        # Get unique cell types
        unique_types = sorted(list(set(cell_types)))
        
        # Create interaction matrix
        interaction_matrix = pd.DataFrame(
            np.zeros((len(unique_types), len(unique_types))),
            index=unique_types,
            columns=unique_types
        )
        
        # Process interactions
        if 'interactions' in interactions:
            for interaction in interactions['interactions']:
                cell_i = interaction['cell_i']
                cell_j = interaction['cell_j']
                confidence = interaction.get('confidence', 1.0)
                
                # Skip if indices out of bounds
                if cell_i >= len(cell_types) or cell_j >= len(cell_types):
                    continue
                
                type_i = cell_types[cell_i]
                type_j = cell_types[cell_j]
                
                # Add to matrix (symmetric)
                if aggregation == "count":
                    interaction_matrix.loc[type_i, type_j] += 1
                    if type_i != type_j:
                        interaction_matrix.loc[type_j, type_i] += 1
                elif aggregation == "sum_confidence":
                    interaction_matrix.loc[type_i, type_j] += confidence
                    if type_i != type_j:
                        interaction_matrix.loc[type_j, type_i] += confidence
                elif aggregation == "mean_confidence":
                    # For mean, we'll compute this after counting
                    interaction_matrix.loc[type_i, type_j] += confidence
                    if type_i != type_j:
                        interaction_matrix.loc[type_j, type_i] += confidence
        
        # Filter cell types with minimum interactions
        row_sums = interaction_matrix.sum(axis=1)
        valid_types = row_sums[row_sums >= min_interactions].index.tolist()
        interaction_matrix = interaction_matrix.loc[valid_types, valid_types]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(8, len(valid_types) * 0.6), 
                                       max(6, len(valid_types) * 0.6)), 
                               dpi=self.dpi)
        
        # Create heatmap
        mask = interaction_matrix == 0
        sns.heatmap(
            interaction_matrix,
            annot=True,
            fmt='.1f' if aggregation != 'count' else '.0f',
            cmap='Reds',
            mask=mask,
            square=True,
            cbar_kws={'label': aggregation.replace('_', ' ').title()},
            ax=ax
        )
        
        # Formatting
        ax.set_xlabel('Cell Type', fontsize=12)
        ax.set_ylabel('Cell Type', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'Cell Type Interaction Matrix ({aggregation})', fontsize=14)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_interactive_network(
        self,
        interactions: Dict[str, Any],
        cell_types: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        max_interactions: int = 200,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive network plot using Plotly.
        
        Args:
            interactions: Interaction predictions
            cell_types: List of cell type names
            confidence_threshold: Minimum confidence threshold
            max_interactions: Maximum interactions to display
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        # Create networkx graph
        G = nx.Graph()
        
        # Filter interactions
        if 'interactions' not in interactions:
            raise ValueError("Interactions dictionary must contain 'interactions' key")
        
        interaction_list = interactions['interactions']
        filtered_interactions = [
            inter for inter in interaction_list
            if inter.get('confidence', 0) >= confidence_threshold
        ][:max_interactions]
        
        # Add nodes and edges
        for interaction in filtered_interactions:
            cell_i = interaction['cell_i']
            cell_j = interaction['cell_j']
            confidence = interaction['confidence']
            G.add_edge(cell_i, cell_j, weight=confidence)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            confidence = G[edge[0]][edge[1]]['weight']
            edge_info.append(f"Confidence: {confidence:.3f}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        
        # Color mapping for cell types
        if cell_types:
            unique_types = list(set(cell_types))
            colors = px.colors.qualitative.Set3[:len(unique_types)]
            color_map = dict(zip(unique_types, colors))
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node info
            adjacencies = list(G.neighbors(node))
            node_info = f'Cell {node}<br>'
            node_info += f'# of connections: {len(adjacencies)}<br>'
            
            if cell_types and node < len(cell_types):
                cell_type = cell_types[node]
                node_info += f'Cell type: {cell_type}'
                node_colors.append(color_map.get(cell_type, '#888'))
            else:
                node_colors.append('#888')
            
            node_text.append(node_info)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=node_colors,
                size=10,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2, color='black')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title or f'Interactive Cell-Cell Interaction Network (n={len(filtered_interactions)})',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[dict(
                    text="Hover over nodes for details",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='#888', size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        )
        
        return fig
    
    def create_interaction_summary(
        self,
        interactions: Dict[str, Any],
        dataset: Optional[BaseSpatialDataset] = None,
        cell_types: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive interaction summary with multiple panels.
        
        Args:
            interactions: Interaction predictions
            dataset: Optional spatial dataset
            cell_types: List of cell type names
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure with multiple panels
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12), dpi=self.dpi)
        
        # Panel 1: Network graph (top-left)
        ax1 = plt.subplot(2, 3, 1)
        self._plot_network_panel(ax1, interactions, cell_types)
        
        # Panel 2: Spatial interactions if dataset available (top-middle)
        ax2 = plt.subplot(2, 3, 2)
        if dataset:
            self._plot_spatial_panel(ax2, dataset, interactions)
        else:
            ax2.text(0.5, 0.5, 'No spatial data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Spatial Interactions')
        
        # Panel 3: Interaction heatmap (top-right)
        ax3 = plt.subplot(2, 3, 3)
        if cell_types:
            self._plot_heatmap_panel(ax3, interactions, cell_types)
        else:
            ax3.text(0.5, 0.5, 'No cell type data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Interaction Heatmap')
        
        # Panel 4: Confidence distribution (bottom-left)
        ax4 = plt.subplot(2, 3, 4)
        self._plot_confidence_distribution(ax4, interactions)
        
        # Panel 5: Distance distribution (bottom-middle)
        ax5 = plt.subplot(2, 3, 5)
        if dataset:
            self._plot_distance_distribution(ax5, interactions, dataset)
        else:
            ax5.text(0.5, 0.5, 'No spatial data available', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Distance Distribution')
        
        # Panel 6: Statistics (bottom-right)
        ax6 = plt.subplot(2, 3, 6)
        self._plot_statistics_panel(ax6, interactions, dataset, cell_types)
        
        plt.suptitle('Cell-Cell Interaction Analysis Summary', fontsize=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _plot_network_panel(
        self,
        ax: plt.Axes,
        interactions: Dict[str, Any],
        cell_types: Optional[List[str]] = None
    ) -> None:
        """Plot network graph panel."""
        # Simple network visualization
        G = nx.Graph()
        
        if 'interactions' in interactions:
            for interaction in interactions['interactions'][:50]:  # Limit for panel
                cell_i = interaction['cell_i']
                cell_j = interaction['cell_j']
                confidence = interaction['confidence']
                G.add_edge(cell_i, cell_j, weight=confidence)
        
        if G.nodes():
            pos = nx.spring_layout(G, k=0.5)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50, alpha=0.7)
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, width=0.5)
        
        ax.set_title('Interaction Network')
        ax.axis('off')
    
    def _plot_spatial_panel(
        self,
        ax: plt.Axes,
        dataset: BaseSpatialDataset,
        interactions: Dict[str, Any]
    ) -> None:
        """Plot spatial interactions panel."""
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        ax.scatter(spatial_coords[:, 0], spatial_coords[:, 1], 
                  c='lightgray', s=5, alpha=0.6)
        
        if 'interactions' in interactions:
            for interaction in interactions['interactions'][:20]:
                cell_i = interaction['cell_i']
                cell_j = interaction['cell_j']
                confidence = interaction['confidence']
                
                if cell_i < len(spatial_coords) and cell_j < len(spatial_coords):
                    ax.plot([spatial_coords[cell_i, 0], spatial_coords[cell_j, 0]],
                           [spatial_coords[cell_i, 1], spatial_coords[cell_j, 1]],
                           color='red', alpha=confidence, linewidth=0.5)
        
        ax.set_title('Spatial Interactions')
        ax.set_aspect('equal')
    
    def _plot_heatmap_panel(
        self,
        ax: plt.Axes,
        interactions: Dict[str, Any],
        cell_types: List[str]
    ) -> None:
        """Plot interaction heatmap panel."""
        # Simplified heatmap for panel
        unique_types = sorted(list(set(cell_types)))[:8]  # Limit for readability
        matrix = pd.DataFrame(
            np.zeros((len(unique_types), len(unique_types))),
            index=unique_types, columns=unique_types
        )
        
        if 'interactions' in interactions:
            for interaction in interactions['interactions']:
                cell_i = interaction['cell_i']
                cell_j = interaction['cell_j']
                
                if cell_i < len(cell_types) and cell_j < len(cell_types):
                    type_i = cell_types[cell_i]
                    type_j = cell_types[cell_j]
                    
                    if type_i in unique_types and type_j in unique_types:
                        matrix.loc[type_i, type_j] += 1
        
        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Reds', ax=ax, 
                   cbar=False, square=True)
        ax.set_title('Cell Type Interactions')
    
    def _plot_confidence_distribution(
        self,
        ax: plt.Axes,
        interactions: Dict[str, Any]
    ) -> None:
        """Plot confidence distribution."""
        if 'interactions' in interactions:
            confidences = [inter['confidence'] for inter in interactions['interactions']]
            ax.hist(confidences, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(confidences):.3f}')
            ax.legend()
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Distribution')
    
    def _plot_distance_distribution(
        self,
        ax: plt.Axes,
        interactions: Dict[str, Any],
        dataset: BaseSpatialDataset
    ) -> None:
        """Plot interaction distance distribution."""
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        distances = []
        if 'interactions' in interactions:
            for interaction in interactions['interactions']:
                cell_i = interaction['cell_i']
                cell_j = interaction['cell_j']
                
                if cell_i < len(spatial_coords) and cell_j < len(spatial_coords):
                    dist = np.linalg.norm(
                        spatial_coords[cell_i] - spatial_coords[cell_j]
                    )
                    distances.append(dist)
        
        if distances:
            ax.hist(distances, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax.axvline(np.mean(distances), color='red', linestyle='--',
                      label=f'Mean: {np.mean(distances):.1f}μm')
            ax.legend()
        
        ax.set_xlabel('Distance (μm)')
        ax.set_ylabel('Count')
        ax.set_title('Interaction Distance Distribution')
    
    def _plot_statistics_panel(
        self,
        ax: plt.Axes,
        interactions: Dict[str, Any],
        dataset: Optional[BaseSpatialDataset] = None,
        cell_types: Optional[List[str]] = None
    ) -> None:
        """Plot statistics panel."""
        ax.axis('off')
        
        stats_text = "Interaction Statistics:\n\n"
        
        if 'interactions' in interactions:
            n_interactions = len(interactions['interactions'])
            stats_text += f"• Total interactions: {n_interactions:,}\n"
            
            confidences = [inter['confidence'] for inter in interactions['interactions']]
            stats_text += f"• Mean confidence: {np.mean(confidences):.3f}\n"
            stats_text += f"• Std confidence: {np.std(confidences):.3f}\n\n"
        
        if dataset:
            stats_text += f"• Total cells: {dataset.num_cells:,}\n"
            stats_text += f"• Interaction rate: {n_interactions/dataset.num_cells:.3f}\n\n"
        
        if cell_types:
            unique_types = set(cell_types)
            stats_text += f"• Cell types: {len(unique_types)}\n"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        ax.set_title('Statistics')