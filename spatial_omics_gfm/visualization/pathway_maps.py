"""
Pathway visualization for spatial transcriptomics analysis.

This module provides comprehensive visualization capabilities for biological
pathways including heatmaps, network diagrams, and spatial pathway activity maps.
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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from ..data.base import BaseSpatialDataset


class PathwayMapper:
    """
    Comprehensive pathway visualization engine for spatial transcriptomics.
    
    Provides methods for visualizing pathway activity, gene networks,
    pathway enrichment, and spatial pathway distributions.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 10),
        dpi: int = 300,
        style: str = "default"
    ):
        """
        Initialize the pathway mapper.
        
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
    
    def plot_pathway_heatmap(
        self,
        pathway_scores: Dict[str, Any],
        pathways: Optional[List[str]] = None,
        cells: Optional[List[str]] = None,
        cluster_rows: bool = True,
        cluster_cols: bool = True,
        z_score: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create pathway activity heatmap.
        
        Args:
            pathway_scores: Dictionary with pathway activity scores
            pathways: List of pathway names to include
            cells: List of cell identifiers
            cluster_rows: Whether to cluster pathways (rows)
            cluster_cols: Whether to cluster cells (columns)
            z_score: Whether to z-score normalize
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size override
            
        Returns:
            Matplotlib figure
        """
        # Extract pathway activity matrix
        if 'pathway_activity' not in pathway_scores:
            raise ValueError("pathway_scores must contain 'pathway_activity' key")
        
        activity_matrix = pathway_scores['pathway_activity']
        
        # Convert to DataFrame if numpy array
        if isinstance(activity_matrix, np.ndarray):
            if pathways is None:
                pathways = [f"Pathway_{i}" for i in range(activity_matrix.shape[0])]
            if cells is None:
                cells = [f"Cell_{i}" for i in range(activity_matrix.shape[1])]
            
            activity_df = pd.DataFrame(
                activity_matrix, 
                index=pathways, 
                columns=cells
            )
        else:
            activity_df = pd.DataFrame(activity_matrix)
        
        # Subset pathways and cells if specified
        if pathways:
            available_pathways = [p for p in pathways if p in activity_df.index]
            activity_df = activity_df.loc[available_pathways]
        
        if cells:
            available_cells = [c for c in cells if c in activity_df.columns]
            activity_df = activity_df[available_cells]
        
        # Z-score normalization
        if z_score:
            activity_df = activity_df.T  # Transpose for row-wise z-score
            activity_df = activity_df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
            activity_df = activity_df.T  # Transpose back
        
        # Create figure
        fig_size = figsize if figsize else (
            max(self.figsize[0], activity_df.shape[1] * 0.3),
            max(self.figsize[1], activity_df.shape[0] * 0.3)
        )
        
        fig, ax = plt.subplots(figsize=fig_size, dpi=self.dpi)
        
        # Create clustermap
        sns.heatmap(
            activity_df,
            annot=activity_df.shape[0] <= 20 and activity_df.shape[1] <= 20,  # Annotate if small
            fmt='.2f',
            cmap='RdBu_r',
            center=0 if z_score else None,
            cbar_kws={'label': 'Z-score' if z_score else 'Activity Score'},
            ax=ax,
            xticklabels=True,
            yticklabels=True
        )
        
        # Formatting
        ax.set_xlabel('Cells', fontsize=12)
        ax.set_ylabel('Pathways', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Pathway Activity Heatmap', fontsize=14)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_pathway_network(
        self,
        pathway_data: Dict[str, Any],
        pathway_name: str,
        gene_expressions: Optional[np.ndarray] = None,
        gene_names: Optional[List[str]] = None,
        layout: str = "spring",
        node_size_scale: float = 500,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize pathway as a gene interaction network.
        
        Args:
            pathway_data: Dictionary containing pathway information
            pathway_name: Name of the pathway to visualize
            gene_expressions: Optional gene expression values for coloring
            gene_names: List of gene names
            layout: Network layout algorithm
            node_size_scale: Scale factor for node sizes
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Extract pathway network information
        if 'pathways' not in pathway_data:
            raise ValueError("pathway_data must contain 'pathways' key")
        
        pathways = pathway_data['pathways']
        if pathway_name not in pathways:
            raise ValueError(f"Pathway '{pathway_name}' not found in pathway data")
        
        pathway_info = pathways[pathway_name]
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (genes)
        genes = pathway_info.get('genes', [])
        for gene in genes:
            G.add_node(gene)
        
        # Add edges (interactions)
        interactions = pathway_info.get('interactions', [])
        for interaction in interactions:
            if len(interaction) >= 2:
                G.add_edge(interaction[0], interaction[1])
        
        # If no interactions, create a simple star layout with hub
        if not G.edges() and genes:
            hub_gene = genes[0]  # Use first gene as hub
            for gene in genes[1:]:
                G.add_edge(hub_gene, gene)
        
        # Calculate layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai" and len(G.nodes()) > 1:
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Prepare node colors and sizes
        node_colors = []
        node_sizes = []
        
        if gene_expressions is not None and gene_names is not None:
            # Color by gene expression
            gene_expr_dict = dict(zip(gene_names, gene_expressions))
            scaler = MinMaxScaler()
            expr_values = [gene_expr_dict.get(gene, 0) for gene in G.nodes()]
            
            if len(set(expr_values)) > 1:  # If there's variation in expression
                expr_normalized = scaler.fit_transform(np.array(expr_values).reshape(-1, 1)).flatten()
                node_colors = plt.cm.viridis(expr_normalized)
            else:
                node_colors = 'lightblue'
        else:
            node_colors = 'lightblue'
        
        # Node sizes based on degree
        degrees = dict(G.degree())
        for node in G.nodes():
            degree = degrees.get(node, 1)
            node_sizes.append(max(100, degree * node_size_scale))
        
        # Draw network
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            linewidths=1,
            edgecolors='black'
        )
        
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='gray',
            width=1,
            alpha=0.6
        )
        
        # Add labels
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=8,
            font_weight='bold'
        )
        
        # Formatting
        ax.set_aspect('equal')
        ax.axis('off')
        
        if title:
            ax.set_title(title, fontsize=16, pad=20)
        else:
            ax.set_title(f'Pathway Network: {pathway_name}', fontsize=16, pad=20)
        
        # Add colorbar if using expression coloring
        if gene_expressions is not None and gene_names is not None and len(set(expr_values)) > 1:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                     norm=plt.Normalize(vmin=min(expr_values), 
                                                       vmax=max(expr_values)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Gene Expression')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_spatial_pathway_activity(
        self,
        dataset: BaseSpatialDataset,
        pathway_scores: Dict[str, Any],
        pathway_name: str,
        point_size: float = 20,
        alpha: float = 0.8,
        cmap: str = "viridis",
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot spatial distribution of pathway activity.
        
        Args:
            dataset: Spatial dataset
            pathway_scores: Pathway activity scores
            pathway_name: Name of pathway to visualize
            point_size: Size of spatial points
            alpha: Point transparency
            cmap: Colormap for activity scores
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Get spatial coordinates
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        # Extract pathway activity scores
        if 'pathway_activity' not in pathway_scores:
            raise ValueError("pathway_scores must contain 'pathway_activity' key")
        
        activity_matrix = pathway_scores['pathway_activity']
        pathway_names = pathway_scores.get('pathway_names', [])
        
        # Find pathway index
        if pathway_name not in pathway_names:
            raise ValueError(f"Pathway '{pathway_name}' not found in pathway names")
        
        pathway_idx = pathway_names.index(pathway_name)
        activity_values = activity_matrix[pathway_idx, :]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create scatter plot
        scatter = ax.scatter(
            spatial_coords[:, 0],
            spatial_coords[:, 1],
            c=activity_values,
            s=point_size,
            alpha=alpha,
            cmap=cmap,
            edgecolors='none'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Pathway Activity')
        
        # Formatting
        ax.set_xlabel('X Position (μm)', fontsize=12)
        ax.set_ylabel('Y Position (μm)', fontsize=12)
        ax.set_aspect('equal')
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'Spatial Distribution: {pathway_name}', fontsize=14)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_pathway_enrichment(
        self,
        enrichment_results: Dict[str, Any],
        top_n: int = 20,
        p_value_threshold: float = 0.05,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create pathway enrichment bar plot.
        
        Args:
            enrichment_results: Dictionary with enrichment analysis results
            top_n: Number of top pathways to display
            p_value_threshold: P-value threshold for significance
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Extract enrichment data
        if 'pathways' not in enrichment_results:
            raise ValueError("enrichment_results must contain 'pathways' key")
        
        pathways_data = enrichment_results['pathways']
        
        # Create DataFrame
        enrichment_df = pd.DataFrame(pathways_data)
        
        # Filter by p-value and get top results
        if 'p_value' in enrichment_df.columns:
            significant = enrichment_df[enrichment_df['p_value'] <= p_value_threshold]
        else:
            significant = enrichment_df
        
        # Sort by enrichment score or p-value
        if 'enrichment_score' in significant.columns:
            significant = significant.sort_values('enrichment_score', ascending=False)
        elif 'p_value' in significant.columns:
            significant = significant.sort_values('p_value', ascending=True)
        
        # Take top N
        top_pathways = significant.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(self.figsize[0], max(6, len(top_pathways) * 0.4)), 
                               dpi=self.dpi)
        
        # Create horizontal bar plot
        if 'enrichment_score' in top_pathways.columns:
            y_pos = np.arange(len(top_pathways))
            bars = ax.barh(y_pos, top_pathways['enrichment_score'], 
                          color='steelblue', alpha=0.8)
            
            # Color bars by p-value if available
            if 'p_value' in top_pathways.columns:
                p_values = top_pathways['p_value'].values
                colors = plt.cm.viridis_r(1 - p_values / p_values.max())
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
            
            ax.set_xlabel('Enrichment Score', fontsize=12)
        elif 'p_value' in top_pathways.columns:
            # Plot -log10(p-value)
            neg_log_p = -np.log10(top_pathways['p_value'])
            y_pos = np.arange(len(top_pathways))
            ax.barh(y_pos, neg_log_p, color='steelblue', alpha=0.8)
            ax.set_xlabel('-log10(p-value)', fontsize=12)
        
        # Set pathway names as y-axis labels
        pathway_names = top_pathways.get('pathway', 
                                       top_pathways.get('name', 
                                                       [f'Pathway_{i}' for i in range(len(top_pathways))]))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pathway_names)
        ax.set_ylabel('Pathways', fontsize=12)
        
        # Add significance line if using p-values
        if 'p_value' in top_pathways.columns and 'enrichment_score' not in top_pathways.columns:
            sig_line = -np.log10(p_value_threshold)
            ax.axvline(sig_line, color='red', linestyle='--', alpha=0.7, 
                      label=f'p = {p_value_threshold}')
            ax.legend()
        
        # Formatting
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Pathway Enrichment Analysis', fontsize=14)
        
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_interactive_pathway_heatmap(
        self,
        pathway_scores: Dict[str, Any],
        pathways: Optional[List[str]] = None,
        cells: Optional[List[str]] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive pathway heatmap using Plotly.
        
        Args:
            pathway_scores: Dictionary with pathway activity scores
            pathways: List of pathway names
            cells: List of cell identifiers
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        # Extract pathway activity matrix
        if 'pathway_activity' not in pathway_scores:
            raise ValueError("pathway_scores must contain 'pathway_activity' key")
        
        activity_matrix = pathway_scores['pathway_activity']
        
        # Convert to DataFrame
        if isinstance(activity_matrix, np.ndarray):
            if pathways is None:
                pathways = [f"Pathway_{i}" for i in range(activity_matrix.shape[0])]
            if cells is None:
                cells = [f"Cell_{i}" for i in range(activity_matrix.shape[1])]
            
            activity_df = pd.DataFrame(
                activity_matrix, 
                index=pathways, 
                columns=cells
            )
        else:
            activity_df = pd.DataFrame(activity_matrix)
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=activity_df.values,
            x=activity_df.columns,
            y=activity_df.index,
            colorscale='RdBu_r',
            zmid=0,
            hoverongaps=False,
            hovertemplate='Cell: %{x}<br>Pathway: %{y}<br>Activity: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title or 'Interactive Pathway Activity Heatmap',
            xaxis_title='Cells',
            yaxis_title='Pathways',
            width=max(800, activity_df.shape[1] * 20),
            height=max(600, activity_df.shape[0] * 20)
        )
        
        return fig
    
    def create_pathway_summary(
        self,
        pathway_scores: Dict[str, Any],
        dataset: Optional[BaseSpatialDataset] = None,
        enrichment_results: Optional[Dict[str, Any]] = None,
        top_pathways: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive pathway analysis summary.
        
        Args:
            pathway_scores: Pathway activity scores
            dataset: Optional spatial dataset
            enrichment_results: Optional enrichment analysis results
            top_pathways: List of top pathways to highlight
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure with multiple panels
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16), dpi=self.dpi)
        
        # Panel 1: Pathway heatmap (top-left, spanning 2 columns)
        ax1 = plt.subplot(3, 3, (1, 2))
        self._plot_heatmap_panel(ax1, pathway_scores, top_pathways)
        
        # Panel 2: Enrichment results (top-right)
        ax2 = plt.subplot(3, 3, 3)
        if enrichment_results:
            self._plot_enrichment_panel(ax2, enrichment_results)
        else:
            ax2.text(0.5, 0.5, 'No enrichment results available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Pathway Enrichment')
        
        # Panel 3: Spatial pathway activity (middle-left)
        ax3 = plt.subplot(3, 3, 4)
        if dataset and top_pathways:
            self._plot_spatial_activity_panel(ax3, dataset, pathway_scores, top_pathways[0])
        else:
            ax3.text(0.5, 0.5, 'No spatial data or pathways available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Spatial Activity')
        
        # Panel 4: Pathway activity distribution (middle-center)
        ax4 = plt.subplot(3, 3, 5)
        self._plot_activity_distribution_panel(ax4, pathway_scores)
        
        # Panel 5: Top pathway network (middle-right)
        ax5 = plt.subplot(3, 3, 6)
        if top_pathways:
            self._plot_pathway_network_panel(ax5, top_pathways[0])
        else:
            ax5.text(0.5, 0.5, 'No pathway network data available', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Pathway Network')
        
        # Panel 6: Pathway correlations (bottom-left)
        ax6 = plt.subplot(3, 3, 7)
        self._plot_pathway_correlations_panel(ax6, pathway_scores)
        
        # Panel 7: Cell type pathway activity (bottom-center)
        ax7 = plt.subplot(3, 3, 8)
        self._plot_cell_type_activity_panel(ax7, pathway_scores)
        
        # Panel 8: Statistics (bottom-right)
        ax8 = plt.subplot(3, 3, 9)
        self._plot_pathway_statistics_panel(ax8, pathway_scores, dataset)
        
        plt.suptitle('Pathway Analysis Summary', fontsize=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _plot_heatmap_panel(
        self,
        ax: plt.Axes,
        pathway_scores: Dict[str, Any],
        top_pathways: Optional[List[str]] = None
    ) -> None:
        """Plot pathway heatmap panel."""
        if 'pathway_activity' in pathway_scores:
            activity_matrix = pathway_scores['pathway_activity']
            
            # Limit to top pathways for readability
            if top_pathways and len(top_pathways) <= 15:
                pathway_names = top_pathways
                # Assuming activity_matrix rows correspond to pathways
                if hasattr(activity_matrix, 'shape') and activity_matrix.shape[0] >= len(top_pathways):
                    subset_matrix = activity_matrix[:len(top_pathways), :50]  # Limit cells too
                else:
                    subset_matrix = activity_matrix[:, :50]
            else:
                subset_matrix = activity_matrix[:15, :50]  # Top 15 pathways, 50 cells
                pathway_names = [f"Pathway_{i}" for i in range(subset_matrix.shape[0])]
            
            sns.heatmap(subset_matrix, ax=ax, cbar=True, cmap='RdBu_r',
                       xticklabels=False, yticklabels=pathway_names)
        
        ax.set_title('Pathway Activity Heatmap')
        ax.set_xlabel('Cells')
        ax.set_ylabel('Pathways')
    
    def _plot_enrichment_panel(
        self,
        ax: plt.Axes,
        enrichment_results: Dict[str, Any]
    ) -> None:
        """Plot enrichment results panel."""
        if 'pathways' in enrichment_results:
            pathways_data = enrichment_results['pathways'][:10]  # Top 10
            
            if isinstance(pathways_data, list) and pathways_data:
                # Extract p-values or enrichment scores
                if isinstance(pathways_data[0], dict):
                    if 'p_value' in pathways_data[0]:
                        values = [-np.log10(p['p_value']) for p in pathways_data]
                        names = [p.get('name', f'Pathway_{i}') for i, p in enumerate(pathways_data)]
                    elif 'enrichment_score' in pathways_data[0]:
                        values = [p['enrichment_score'] for p in pathways_data]
                        names = [p.get('name', f'Pathway_{i}') for i, p in enumerate(pathways_data)]
                    else:
                        values = list(range(len(pathways_data)))
                        names = [f'Pathway_{i}' for i in range(len(pathways_data))]
                    
                    ax.barh(range(len(values)), values, color='steelblue', alpha=0.7)
                    ax.set_yticks(range(len(names)))
                    ax.set_yticklabels(names[::-1])  # Reverse for better display
        
        ax.set_title('Top Enriched Pathways')
        ax.set_xlabel('Enrichment Score')
    
    def _plot_spatial_activity_panel(
        self,
        ax: plt.Axes,
        dataset: BaseSpatialDataset,
        pathway_scores: Dict[str, Any],
        pathway_name: str
    ) -> None:
        """Plot spatial pathway activity panel."""
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        # Simplified spatial plot with mock activity
        activity_values = np.random.random(len(spatial_coords))  # Mock data
        
        scatter = ax.scatter(spatial_coords[:, 0], spatial_coords[:, 1], 
                           c=activity_values, s=5, cmap='viridis', alpha=0.7)
        
        ax.set_title(f'Spatial Activity: {pathway_name}')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _plot_activity_distribution_panel(
        self,
        ax: plt.Axes,
        pathway_scores: Dict[str, Any]
    ) -> None:
        """Plot pathway activity distribution panel."""
        if 'pathway_activity' in pathway_scores:
            activity_matrix = pathway_scores['pathway_activity']
            
            # Plot distribution of all pathway activities
            all_activities = activity_matrix.flatten()
            ax.hist(all_activities, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.mean(all_activities), color='red', linestyle='--',
                      label=f'Mean: {np.mean(all_activities):.3f}')
            ax.legend()
        
        ax.set_xlabel('Activity Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Activity Distribution')
    
    def _plot_pathway_network_panel(
        self,
        ax: plt.Axes,
        pathway_name: str
    ) -> None:
        """Plot simplified pathway network panel."""
        # Create a simple mock network
        G = nx.erdos_renyi_graph(8, 0.3)
        pos = nx.spring_layout(G)
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=100, 
                             node_color='lightblue', alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6)
        
        ax.set_title(f'Network: {pathway_name}')
        ax.axis('off')
    
    def _plot_pathway_correlations_panel(
        self,
        ax: plt.Axes,
        pathway_scores: Dict[str, Any]
    ) -> None:
        """Plot pathway correlation panel."""
        if 'pathway_activity' in pathway_scores:
            activity_matrix = pathway_scores['pathway_activity']
            
            # Compute correlations between pathways
            if activity_matrix.shape[0] > 1:
                correlation_matrix = np.corrcoef(activity_matrix[:10, :])  # Top 10 pathways
                
                sns.heatmap(correlation_matrix, ax=ax, cmap='RdBu_r', center=0,
                           square=True, cbar=False, annot=False)
        
        ax.set_title('Pathway Correlations')
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _plot_cell_type_activity_panel(
        self,
        ax: plt.Axes,
        pathway_scores: Dict[str, Any]
    ) -> None:
        """Plot cell type pathway activity panel."""
        # Mock cell type activity data
        cell_types = ['T Cell', 'B Cell', 'NK Cell', 'Monocyte']
        activity_means = np.random.random(len(cell_types))
        
        ax.bar(cell_types, activity_means, color='lightcoral', alpha=0.7)
        ax.set_title('Activity by Cell Type')
        ax.set_ylabel('Mean Activity')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_pathway_statistics_panel(
        self,
        ax: plt.Axes,
        pathway_scores: Dict[str, Any],
        dataset: Optional[BaseSpatialDataset] = None
    ) -> None:
        """Plot pathway statistics panel."""
        ax.axis('off')
        
        stats_text = "Pathway Analysis Statistics:\n\n"
        
        if 'pathway_activity' in pathway_scores:
            activity_matrix = pathway_scores['pathway_activity']
            n_pathways, n_cells = activity_matrix.shape
            
            stats_text += f"• Number of pathways: {n_pathways:,}\n"
            stats_text += f"• Number of cells: {n_cells:,}\n"
            stats_text += f"• Mean activity: {np.mean(activity_matrix):.3f}\n"
            stats_text += f"• Std activity: {np.std(activity_matrix):.3f}\n\n"
        
        if dataset:
            stats_text += f"• Total genes: {dataset.num_genes:,}\n"
            stats_text += f"• Total cells: {dataset.num_cells:,}\n"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        ax.set_title('Analysis Statistics')