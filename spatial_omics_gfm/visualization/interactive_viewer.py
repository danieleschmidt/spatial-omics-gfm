"""
Interactive web-based viewer for spatial transcriptomics analysis.

This module provides comprehensive interactive visualization capabilities
using Plotly and Dash for real-time exploration of spatial omics data.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any, Union, Tuple, Callable
import json
from pathlib import Path
import warnings

try:
    import dash
    from dash import dcc, html, Input, Output, State, callback_context
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    warnings.warn("Dash not available. Interactive dashboard features will be limited.")

from ..data.base import BaseSpatialDataset


class InteractiveSpatialViewer:
    """
    Interactive viewer for spatial transcriptomics data.
    
    Provides web-based interactive visualizations with real-time filtering,
    zoom, hover information, and multi-panel comparisons using Plotly and Dash.
    """
    
    def __init__(
        self,
        dataset: Optional[BaseSpatialDataset] = None,
        width: int = 1200,
        height: int = 800,
        theme: str = "plotly_white"
    ):
        """
        Initialize the interactive viewer.
        
        Args:
            dataset: Optional spatial dataset to load initially
            width: Default plot width
            height: Default plot height
            theme: Plotly theme ('plotly', 'plotly_white', 'plotly_dark', 'ggplot2')
        """
        self.dataset = dataset
        self.width = width
        self.height = height
        self.theme = theme
        
        # Cache for data
        self.cached_data = {}
        self.cached_predictions = {}
        self.cached_interactions = {}
        self.cached_pathways = {}
        
        # Default color palettes
        self.color_palettes = {
            'categorical': px.colors.qualitative.Set3,
            'continuous': px.colors.sequential.Viridis,
            'diverging': px.colors.diverging.RdBu
        }
    
    def create_interactive_spatial_plot(
        self,
        dataset: Optional[BaseSpatialDataset] = None,
        predictions: Optional[Dict[str, Any]] = None,
        color_by: str = "cell_type",
        gene_name: Optional[str] = None,
        point_size: float = 5,
        opacity: float = 0.7,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive spatial plot with hover information.
        
        Args:
            dataset: Spatial dataset to plot
            predictions: Model predictions
            color_by: What to color points by
            gene_name: Gene name for expression coloring
            point_size: Size of points
            opacity: Point opacity
            title: Plot title
            
        Returns:
            Interactive Plotly figure
        """
        if dataset is None:
            dataset = self.dataset
            
        if dataset is None:
            raise ValueError("No dataset provided")
        
        # Get data
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        # Prepare hover data
        hover_data = {
            'x': spatial_coords[:, 0],
            'y': spatial_coords[:, 1],
            'cell_id': list(range(len(spatial_coords)))
        }
        
        # Determine colors and prepare data
        if color_by == "cell_type" and hasattr(data, 'cell_type'):
            colors = data.cell_type.numpy() if hasattr(data.cell_type, 'numpy') else data.cell_type
            hover_data['cell_type'] = colors
            color_column = 'cell_type'
            is_continuous = False
            
        elif color_by == "gene" and gene_name:
            if gene_name in dataset.gene_names:
                gene_idx = dataset.gene_names.index(gene_name)
                colors = data.x[:, gene_idx].numpy()
                hover_data['expression'] = colors
                color_column = 'expression'
                is_continuous = True
            else:
                colors = np.zeros(len(spatial_coords))
                hover_data['expression'] = colors
                color_column = 'expression'
                is_continuous = True
                
        elif color_by == "prediction" and predictions and 'predictions' in predictions:
            colors = predictions['predictions']
            hover_data['prediction'] = colors
            color_column = 'prediction'
            is_continuous = False
            
            # Add confidence if available
            if 'confidence' in predictions:
                hover_data['confidence'] = predictions['confidence']
                
        elif color_by == "confidence" and predictions and 'confidence' in predictions:
            colors = predictions['confidence']
            hover_data['confidence'] = colors
            color_column = 'confidence'
            is_continuous = True
            
        else:
            colors = ['Cell'] * len(spatial_coords)
            hover_data['type'] = colors
            color_column = 'type'
            is_continuous = False
        
        # Create DataFrame
        df = pd.DataFrame(hover_data)
        
        # Create scatter plot
        if is_continuous:
            fig = px.scatter(
                df, x='x', y='y', color=color_column,
                color_continuous_scale=self.color_palettes['continuous'],
                size_max=point_size,
                opacity=opacity,
                hover_data={col: True for col in df.columns if col not in ['x', 'y']},
                title=title or f'Interactive Spatial Plot - {color_by.replace("_", " ").title()}'
            )
        else:
            fig = px.scatter(
                df, x='x', y='y', color=color_column,
                color_discrete_sequence=self.color_palettes['categorical'],
                size_max=point_size,
                opacity=opacity,
                hover_data={col: True for col in df.columns if col not in ['x', 'y']},
                title=title or f'Interactive Spatial Plot - {color_by.replace("_", " ").title()}'
            )
        
        # Update layout
        fig.update_layout(
            width=self.width,
            height=self.height,
            template=self.theme,
            xaxis_title="X Position (μm)",
            yaxis_title="Y Position (μm)",
            font=dict(size=12),
            hovermode='closest'
        )
        
        # Ensure equal aspect ratio
        fig.update_yaxis(scaleanchor="x", scaleratio=1)
        
        # Update hover template
        hover_template = "<b>Cell %{customdata[0]}</b><br>"
        hover_template += "X: %{x:.1f} μm<br>"
        hover_template += "Y: %{y:.1f} μm<br>"
        
        if is_continuous:
            hover_template += f"{color_column.title()}: %{{marker.color:.3f}}<br>"
        else:
            hover_template += f"{color_column.title()}: %{{marker.color}}<br>"
        
        hover_template += "<extra></extra>"
        
        fig.update_traces(
            hovertemplate=hover_template,
            customdata=df[['cell_id']].values
        )
        
        return fig
    
    def create_interactive_gene_expression_plot(
        self,
        dataset: Optional[BaseSpatialDataset] = None,
        gene_names: Union[str, List[str]] = [],
        ncols: int = 2,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive multi-gene expression plot.
        
        Args:
            dataset: Spatial dataset
            gene_names: Gene name(s) to plot
            ncols: Number of columns for subplots
            title: Overall title
            
        Returns:
            Interactive Plotly figure with subplots
        """
        if dataset is None:
            dataset = self.dataset
            
        if dataset is None:
            raise ValueError("No dataset provided")
        
        if isinstance(gene_names, str):
            gene_names = [gene_names]
        
        if not gene_names:
            # Use first few genes as default
            gene_names = dataset.gene_names[:4] if hasattr(dataset, 'gene_names') else ['Gene_0', 'Gene_1', 'Gene_2', 'Gene_3']
        
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
        
        # Plot each gene
        for i, gene_name in enumerate(gene_names):
            row = i // ncols + 1
            col = i % ncols + 1
            
            # Get gene expression
            if hasattr(dataset, 'gene_names') and gene_name in dataset.gene_names:
                gene_idx = dataset.gene_names.index(gene_name)
                expression = data.x[:, gene_idx].numpy()
            else:
                expression = np.random.random(len(spatial_coords))  # Mock data
            
            # Create scatter trace
            scatter = go.Scatter(
                x=spatial_coords[:, 0],
                y=spatial_coords[:, 1],
                mode='markers',
                marker=dict(
                    color=expression,
                    colorscale=self.color_palettes['continuous'],
                    size=3,
                    opacity=0.8,
                    colorbar=dict(
                        title=dict(text="Expression", side="right"),
                        x=1.0 + (col - 1) * 0.1,  # Offset colorbars
                        len=0.8 / nrows,
                        y=1 - (row - 1) * (1.0 / nrows) - 0.1 / nrows
                    )
                ),
                hovertemplate=(
                    f"<b>{gene_name}</b><br>"
                    "X: %{x:.1f} μm<br>"
                    "Y: %{y:.1f} μm<br>"
                    "Expression: %{marker.color:.3f}<br>"
                    "<extra></extra>"
                ),
                name=gene_name,
                showlegend=False
            )
            
            fig.add_trace(scatter, row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title=title or "Interactive Gene Expression Patterns",
            width=self.width,
            height=self.height,
            template=self.theme,
            font=dict(size=10)
        )
        
        # Update axes to ensure equal aspect ratio
        for i in range(1, nrows + 1):
            for j in range(1, ncols + 1):
                fig.update_xaxes(title_text="X Position (μm)", row=i, col=j)
                fig.update_yaxes(title_text="Y Position (μm)", scaleanchor=f"x{i}", row=i, col=j)
        
        return fig
    
    def create_interactive_pathway_plot(
        self,
        pathway_scores: Dict[str, Any],
        dataset: Optional[BaseSpatialDataset] = None,
        pathway_name: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive pathway activity visualization.
        
        Args:
            pathway_scores: Pathway activity scores
            dataset: Optional spatial dataset
            pathway_name: Name of pathway to visualize
            title: Plot title
            
        Returns:
            Interactive pathway visualization
        """
        if 'pathway_activity' not in pathway_scores:
            raise ValueError("pathway_scores must contain 'pathway_activity' key")
        
        activity_matrix = pathway_scores['pathway_activity']
        pathway_names = pathway_scores.get('pathway_names', [f'Pathway_{i}' for i in range(activity_matrix.shape[0])])
        
        if pathway_name and pathway_name in pathway_names:
            # Single pathway spatial plot
            if dataset:
                return self._create_spatial_pathway_plot(dataset, pathway_scores, pathway_name, title)
            else:
                raise ValueError("Dataset required for spatial pathway visualization")
        else:
            # Pathway heatmap
            return self._create_interactive_pathway_heatmap(pathway_scores, title)
    
    def create_interactive_interaction_plot(
        self,
        interactions: Dict[str, Any],
        dataset: Optional[BaseSpatialDataset] = None,
        confidence_threshold: float = 0.5,
        max_interactions: int = 100,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive cell-cell interaction visualization.
        
        Args:
            interactions: Interaction predictions
            dataset: Optional spatial dataset
            confidence_threshold: Minimum confidence threshold
            max_interactions: Maximum interactions to display
            title: Plot title
            
        Returns:
            Interactive interaction plot
        """
        if dataset is None:
            dataset = self.dataset
            
        if dataset is None:
            raise ValueError("No dataset provided")
        
        # Get spatial coordinates
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        # Filter interactions
        if 'interactions' not in interactions:
            raise ValueError("interactions must contain 'interactions' key")
        
        interaction_list = interactions['interactions']
        filtered_interactions = [
            inter for inter in interaction_list
            if inter.get('confidence', 0) >= confidence_threshold
        ][:max_interactions]
        
        # Create figure
        fig = go.Figure()
        
        # Add all cells as background
        fig.add_trace(go.Scatter(
            x=spatial_coords[:, 0],
            y=spatial_coords[:, 1],
            mode='markers',
            marker=dict(
                color='lightgray',
                size=4,
                opacity=0.5
            ),
            hovertemplate="Cell %{pointNumber}<br>X: %{x:.1f} μm<br>Y: %{y:.1f} μm<extra></extra>",
            name='All Cells',
            showlegend=True
        ))
        
        # Add interaction edges
        edge_x = []
        edge_y = []
        edge_info = []
        interacting_cells = set()
        
        for interaction in filtered_interactions:
            cell_i = interaction['cell_i']
            cell_j = interaction['cell_j']
            confidence = interaction['confidence']
            
            if cell_i < len(spatial_coords) and cell_j < len(spatial_coords):
                edge_x.extend([spatial_coords[cell_i, 0], spatial_coords[cell_j, 0], None])
                edge_y.extend([spatial_coords[cell_i, 1], spatial_coords[cell_j, 1], None])
                edge_info.append(f"Cells {cell_i}-{cell_j}, Confidence: {confidence:.3f}")
                
                interacting_cells.add(cell_i)
                interacting_cells.add(cell_j)
        
        # Add interaction edges
        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(
                    color='red',
                    width=1
                ),
                hoverinfo='skip',
                name='Interactions',
                showlegend=True
            ))
        
        # Highlight interacting cells
        if interacting_cells:
            interacting_indices = list(interacting_cells)
            fig.add_trace(go.Scatter(
                x=spatial_coords[interacting_indices, 0],
                y=spatial_coords[interacting_indices, 1],
                mode='markers',
                marker=dict(
                    color='red',
                    size=6,
                    opacity=0.8,
                    line=dict(color='darkred', width=1)
                ),
                hovertemplate="Interacting Cell %{pointNumber}<br>X: %{x:.1f} μm<br>Y: %{y:.1f} μm<extra></extra>",
                name='Interacting Cells',
                showlegend=True
            ))
        
        # Update layout
        fig.update_layout(
            title=title or f'Interactive Cell-Cell Interactions (n={len(filtered_interactions)})',
            width=self.width,
            height=self.height,
            template=self.theme,
            xaxis_title="X Position (μm)",
            yaxis_title="Y Position (μm)",
            hovermode='closest'
        )
        
        fig.update_yaxis(scaleanchor="x", scaleratio=1)
        
        return fig
    
    def create_dashboard(
        self,
        dataset: Optional[BaseSpatialDataset] = None,
        predictions: Optional[Dict[str, Any]] = None,
        interactions: Optional[Dict[str, Any]] = None,
        pathway_scores: Optional[Dict[str, Any]] = None,
        port: int = 8050,
        debug: bool = False
    ) -> Any:
        """
        Create interactive dashboard using Dash.
        
        Args:
            dataset: Spatial dataset
            predictions: Model predictions
            interactions: Cell-cell interactions
            pathway_scores: Pathway activity scores
            port: Port for dashboard server
            debug: Whether to run in debug mode
            
        Returns:
            Dash app instance
        """
        if not DASH_AVAILABLE:
            raise ImportError("Dash is required for dashboard functionality. Install with 'pip install dash'")
        
        if dataset is None:
            dataset = self.dataset
        
        # Create Dash app
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Store data
        self.cached_data['dataset'] = dataset
        self.cached_predictions = predictions or {}
        self.cached_interactions = interactions or {}
        self.cached_pathways = pathway_scores or {}
        
        # Define layout
        app.layout = self._create_dashboard_layout(dataset)
        
        # Define callbacks
        self._register_dashboard_callbacks(app)
        
        return app
    
    def _create_spatial_pathway_plot(
        self,
        dataset: BaseSpatialDataset,
        pathway_scores: Dict[str, Any],
        pathway_name: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """Create spatial pathway activity plot."""
        # Get spatial coordinates
        data = dataset.get(0)
        spatial_coords = data.pos.numpy()
        
        # Get pathway activity
        activity_matrix = pathway_scores['pathway_activity']
        pathway_names = pathway_scores.get('pathway_names', [])
        
        if pathway_name in pathway_names:
            pathway_idx = pathway_names.index(pathway_name)
            activity_values = activity_matrix[pathway_idx, :]
        else:
            # Mock activity data
            activity_values = np.random.random(len(spatial_coords))
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=spatial_coords[:, 0],
            y=spatial_coords[:, 1],
            mode='markers',
            marker=dict(
                color=activity_values,
                colorscale=self.color_palettes['continuous'],
                size=5,
                opacity=0.8,
                colorbar=dict(title="Activity Score")
            ),
            hovertemplate=(
                f"<b>{pathway_name} Activity</b><br>"
                "X: %{x:.1f} μm<br>"
                "Y: %{y:.1f} μm<br>"
                "Activity: %{marker.color:.3f}<br>"
                "<extra></extra>"
            ),
            name=pathway_name
        ))
        
        fig.update_layout(
            title=title or f'Spatial Distribution: {pathway_name}',
            width=self.width,
            height=self.height,
            template=self.theme,
            xaxis_title="X Position (μm)",
            yaxis_title="Y Position (μm)"
        )
        
        fig.update_yaxis(scaleanchor="x", scaleratio=1)
        
        return fig
    
    def _create_interactive_pathway_heatmap(
        self,
        pathway_scores: Dict[str, Any],
        title: Optional[str] = None
    ) -> go.Figure:
        """Create interactive pathway heatmap."""
        activity_matrix = pathway_scores['pathway_activity']
        pathway_names = pathway_scores.get('pathway_names', [f'Pathway_{i}' for i in range(activity_matrix.shape[0])])
        
        # Limit size for performance
        max_pathways = 50
        max_cells = 200
        
        if activity_matrix.shape[0] > max_pathways:
            activity_subset = activity_matrix[:max_pathways, :max_cells]
            pathway_subset = pathway_names[:max_pathways]
        else:
            activity_subset = activity_matrix[:, :max_cells]
            pathway_subset = pathway_names
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=activity_subset,
            x=[f'Cell_{i}' for i in range(activity_subset.shape[1])],
            y=pathway_subset,
            colorscale=self.color_palettes['diverging'],
            hoverongaps=False,
            hovertemplate='Pathway: %{y}<br>Cell: %{x}<br>Activity: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title or 'Interactive Pathway Activity Heatmap',
            width=self.width,
            height=self.height,
            template=self.theme,
            xaxis_title="Cells",
            yaxis_title="Pathways"
        )
        
        return fig
    
    def _create_dashboard_layout(self, dataset: Optional[BaseSpatialDataset]) -> html.Div:
        """Create dashboard layout."""
        if not DASH_AVAILABLE:
            return html.Div("Dash not available")
        
        # Get available options
        gene_options = []
        if dataset and hasattr(dataset, 'gene_names'):
            gene_options = [{'label': gene, 'value': gene} for gene in dataset.gene_names[:100]]  # Limit for performance
        
        layout = html.Div([
            dbc.Container([
                # Header
                dbc.Row([
                    dbc.Col([
                        html.H1("Spatial-Omics GFM Interactive Viewer", className="text-center mb-4"),
                        html.Hr()
                    ])
                ]),
                
                # Controls
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Plot Controls"),
                                
                                html.Label("Color by:"),
                                dcc.Dropdown(
                                    id='color-dropdown',
                                    options=[
                                        {'label': 'Cell Type', 'value': 'cell_type'},
                                        {'label': 'Gene Expression', 'value': 'gene'},
                                        {'label': 'Predictions', 'value': 'prediction'},
                                        {'label': 'Confidence', 'value': 'confidence'}
                                    ],
                                    value='cell_type'
                                ),
                                
                                html.Br(),
                                html.Label("Gene (if applicable):"),
                                dcc.Dropdown(
                                    id='gene-dropdown',
                                    options=gene_options,
                                    value=gene_options[0]['value'] if gene_options else None
                                ),
                                
                                html.Br(),
                                html.Label("Point Size:"),
                                dcc.Slider(
                                    id='size-slider',
                                    min=1, max=20, step=1, value=5,
                                    marks={i: str(i) for i in [1, 5, 10, 15, 20]}
                                ),
                                
                                html.Br(),
                                html.Label("Opacity:"),
                                dcc.Slider(
                                    id='opacity-slider',
                                    min=0.1, max=1.0, step=0.1, value=0.7,
                                    marks={i/10: f'{i/10}' for i in range(1, 11)}
                                )
                            ])
                        ])
                    ], width=3),
                    
                    # Main plot
                    dbc.Col([
                        dcc.Graph(
                            id='main-plot',
                            style={'height': '600px'}
                        )
                    ], width=9)
                ]),
                
                html.Hr(),
                
                # Additional plots row
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            id='interaction-plot',
                            style={'height': '400px'}
                        )
                    ], width=6),
                    
                    dbc.Col([
                        dcc.Graph(
                            id='pathway-plot',
                            style={'height': '400px'}
                        )
                    ], width=6)
                ])
            ], fluid=True)
        ])
        
        return layout
    
    def _register_dashboard_callbacks(self, app: Any) -> None:
        """Register dashboard callbacks."""
        if not DASH_AVAILABLE:
            return
        
        @app.callback(
            Output('main-plot', 'figure'),
            [Input('color-dropdown', 'value'),
             Input('gene-dropdown', 'value'),
             Input('size-slider', 'value'),
             Input('opacity-slider', 'value')]
        )
        def update_main_plot(color_by, gene_name, point_size, opacity):
            dataset = self.cached_data.get('dataset')
            predictions = self.cached_predictions
            
            if dataset is None:
                return go.Figure().add_annotation(text="No data available", 
                                                xref="paper", yref="paper",
                                                x=0.5, y=0.5, showarrow=False)
            
            fig = self.create_interactive_spatial_plot(
                dataset=dataset,
                predictions=predictions,
                color_by=color_by,
                gene_name=gene_name,
                point_size=point_size,
                opacity=opacity
            )
            
            return fig
        
        @app.callback(
            Output('interaction-plot', 'figure'),
            [Input('main-plot', 'figure')]  # Trigger update when main plot updates
        )
        def update_interaction_plot(main_fig):
            dataset = self.cached_data.get('dataset')
            interactions = self.cached_interactions
            
            if dataset is None or not interactions:
                return go.Figure().add_annotation(text="No interaction data available",
                                                xref="paper", yref="paper", 
                                                x=0.5, y=0.5, showarrow=False)
            
            fig = self.create_interactive_interaction_plot(
                interactions=interactions,
                dataset=dataset,
                title="Cell-Cell Interactions"
            )
            
            return fig
        
        @app.callback(
            Output('pathway-plot', 'figure'),
            [Input('main-plot', 'figure')]  # Trigger update when main plot updates
        )
        def update_pathway_plot(main_fig):
            pathway_scores = self.cached_pathways
            
            if not pathway_scores:
                return go.Figure().add_annotation(text="No pathway data available",
                                                xref="paper", yref="paper",
                                                x=0.5, y=0.5, showarrow=False)
            
            fig = self.create_interactive_pathway_plot(
                pathway_scores=pathway_scores,
                title="Pathway Activity"
            )
            
            return fig
    
    def export_plot(
        self,
        fig: go.Figure,
        filename: str,
        format: str = "html",
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> None:
        """
        Export interactive plot to file.
        
        Args:
            fig: Plotly figure
            filename: Output filename
            format: Export format ('html', 'png', 'pdf', 'svg')
            width: Output width
            height: Output height
        """
        if format == "html":
            fig.write_html(filename)
        elif format == "png":
            fig.write_image(filename, format='png', width=width, height=height)
        elif format == "pdf":
            fig.write_image(filename, format='pdf', width=width, height=height)
        elif format == "svg":
            fig.write_image(filename, format='svg', width=width, height=height)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def run_dashboard(
        self,
        app: Any,
        host: str = "127.0.0.1",
        port: int = 8050,
        debug: bool = False
    ) -> None:
        """
        Run the dashboard server.
        
        Args:
            app: Dash app instance
            host: Host address
            port: Port number
            debug: Whether to run in debug mode
        """
        if not DASH_AVAILABLE:
            raise ImportError("Dash is required to run dashboard")
        
        print(f"Starting dashboard at http://{host}:{port}")
        app.run_server(host=host, port=port, debug=debug)