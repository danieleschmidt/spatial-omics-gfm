"""
Spatial-aware attention mechanism for Graph Transformers.

This module implements attention mechanisms that explicitly consider
spatial relationships between cells/spots in tissue sections.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Optional, Tuple, Union
import warnings


class SpatialAttention(MessagePassing):
    """
    Spatial-aware attention mechanism for graph neural networks.
    
    This attention mechanism incorporates spatial distance and direction
    information when computing attention weights between cells/spots.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_distance: int = 500,
        dropout: float = 0.1,
        use_edge_features: bool = True,
        bias: bool = True,
        distance_bins: int = 50,
        direction_encoding: bool = True
    ):
        super().__init__(aggr="add", node_dim=0)
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_distance = max_distance
        self.dropout = dropout
        self.use_edge_features = use_edge_features
        self.distance_bins = distance_bins
        self.direction_encoding = direction_encoding
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        # Spatial bias components
        if use_edge_features:
            # Distance embedding
            self.distance_embedding = nn.Embedding(
                distance_bins + 1,  # +1 for out-of-range distances
                num_heads
            )
            
            # Direction encoding (if enabled)
            if direction_encoding:
                self.direction_proj = nn.Sequential(
                    nn.Linear(2, num_heads),  # 2D direction vector
                    nn.Tanh()
                )
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        
        if hasattr(self, 'distance_embedding'):
            nn.init.normal_(self.distance_embedding.weight, std=0.02)
        
        if hasattr(self, 'direction_proj'):
            for layer in self.direction_proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of spatial attention.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
                      Expected format: [distance, dx, dy, ...]
            batch: Batch indices [num_nodes]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Updated node features [num_nodes, hidden_dim]
            Optionally attention weights [num_edges, num_heads]
        """
        # Project to Q, K, V
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        
        # Propagate messages
        out, attention_weights = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            edge_attr=edge_attr,
            batch=batch,
            return_attention_weights=return_attention_weights
        )
        
        # Output projection
        out = out.view(-1, self.hidden_dim)
        out = self.out_proj(out)
        
        if return_attention_weights:
            return out, attention_weights
        return out
    
    def message(
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
        size_i: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute messages for each edge.
        
        Args:
            q_i: Query vectors for target nodes [num_edges, num_heads, head_dim]
            k_j: Key vectors for source nodes [num_edges, num_heads, head_dim]
            v_j: Value vectors for source nodes [num_edges, num_heads, head_dim]
            edge_attr: Edge attributes [num_edges, edge_dim]
            index: Target node indices [num_edges]
            ptr: Batch pointers
            size_i: Number of target nodes
            
        Returns:
            Messages [num_edges, num_heads, head_dim]
        """
        # Compute attention scores
        attention_scores = (q_i * k_j).sum(dim=-1) * self.scale  # [num_edges, num_heads]
        
        # Add spatial bias if edge features are available
        if self.use_edge_features and edge_attr is not None:
            spatial_bias = self._compute_spatial_bias(edge_attr)
            attention_scores = attention_scores + spatial_bias
        
        # Apply softmax normalization
        attention_weights = softmax(attention_scores, index, ptr, size_i)
        attention_weights = self.dropout_layer(attention_weights)
        
        # Store attention weights for visualization/analysis
        self._last_attention_weights = attention_weights
        
        # Apply attention weights to values
        out = attention_weights.unsqueeze(-1) * v_j
        
        return out
    
    def _compute_spatial_bias(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial bias from edge attributes.
        
        Args:
            edge_attr: Edge attributes [num_edges, edge_dim]
                      Expected: [distance, dx, dy, ...]
                      
        Returns:
            Spatial bias [num_edges, num_heads]
        """
        num_edges = edge_attr.size(0)
        spatial_bias = torch.zeros(
            num_edges, self.num_heads, 
            device=edge_attr.device, dtype=edge_attr.dtype
        )
        
        # Distance bias
        if edge_attr.size(1) >= 1:
            distances = edge_attr[:, 0]  # First column is distance
            
            # Bin distances
            distance_bins = torch.clamp(
                (distances / self.max_distance * self.distance_bins).long(),
                max=self.distance_bins
            )
            
            distance_bias = self.distance_embedding(distance_bins)
            spatial_bias += distance_bias
        
        # Direction bias
        if self.direction_encoding and edge_attr.size(1) >= 3:
            directions = edge_attr[:, 1:3]  # dx, dy columns
            
            # Normalize directions
            direction_norms = torch.norm(directions, dim=1, keepdim=True)
            direction_norms = torch.clamp(direction_norms, min=1e-8)
            normalized_directions = directions / direction_norms
            
            direction_bias = self.direction_proj(normalized_directions)
            spatial_bias += direction_bias
        
        return spatial_bias
    
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """
        Update function after message aggregation.
        
        Args:
            aggr_out: Aggregated messages [num_nodes, num_heads, head_dim]
            
        Returns:
            Updated node features [num_nodes, num_heads, head_dim]
        """
        return aggr_out


class MultiScaleSpatialAttention(nn.Module):
    """
    Multi-scale spatial attention that operates at different spatial resolutions.
    
    This module combines attention mechanisms operating at different spatial
    scales to capture both local and global tissue organization patterns.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        scales: list = [50, 200, 500],  # Different distance thresholds
        dropout: float = 0.1,
        combine_method: str = "weighted_sum"
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.scales = scales
        self.num_scales = len(scales)
        self.combine_method = combine_method
        
        # Attention modules for each scale
        self.scale_attentions = nn.ModuleList([
            SpatialAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                max_distance=scale,
                dropout=dropout,
                use_edge_features=True
            )
            for scale in scales
        ])
        
        # Scale combination weights
        if combine_method == "weighted_sum":
            self.scale_weights = nn.Parameter(torch.ones(self.num_scales))
        elif combine_method == "learned_combination":
            self.scale_combiner = nn.Sequential(
                nn.Linear(hidden_dim * self.num_scales, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_indices: list,  # Edge indices for each scale
        edge_attrs: list,    # Edge attributes for each scale
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-scale attention.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_indices: List of edge indices for each scale
            edge_attrs: List of edge attributes for each scale
            batch: Batch indices [num_nodes]
            
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        assert len(edge_indices) == self.num_scales
        assert len(edge_attrs) == self.num_scales
        
        # Compute attention for each scale
        scale_outputs = []
        for i, (attention, edge_index, edge_attr) in enumerate(
            zip(self.scale_attentions, edge_indices, edge_attrs)
        ):
            scale_out = attention(x, edge_index, edge_attr, batch)
            scale_outputs.append(scale_out)
        
        # Combine outputs from different scales
        if self.combine_method == "weighted_sum":
            # Weighted sum with learnable weights
            weights = F.softmax(self.scale_weights, dim=0)
            combined = sum(w * out for w, out in zip(weights, scale_outputs))
            
        elif self.combine_method == "learned_combination":
            # Learned combination via MLP
            concatenated = torch.cat(scale_outputs, dim=-1)
            combined = self.scale_combiner(concatenated)
            
        else:
            # Simple average
            combined = torch.stack(scale_outputs, dim=0).mean(dim=0)
        
        return combined