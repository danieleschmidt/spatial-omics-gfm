"""
Hierarchical pooling for multi-scale spatial analysis.

This module implements pooling operations that create hierarchical
representations of tissue organization at different spatial scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean, scatter_max, scatter_add
from typing import Optional, List, Dict, Tuple, Union
import numpy as np


class HierarchicalPooling(nn.Module):
    """
    Hierarchical pooling for multi-scale spatial analysis.
    
    Creates representations at different spatial scales by grouping
    nearby cells/spots and applying pooling operations.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_scales: int = 4,
        pooling_method: str = "attention",
        base_radius: float = 50.0,
        scale_factor: float = 2.0,
        min_nodes_per_cluster: int = 3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.pooling_method = pooling_method
        self.base_radius = base_radius
        self.scale_factor = scale_factor
        self.min_nodes_per_cluster = min_nodes_per_cluster
        
        # Pooling radius for each scale
        self.radii = [
            base_radius * (scale_factor ** i) 
            for i in range(num_scales)
        ]
        
        # Scale-specific pooling layers
        if pooling_method == "attention":
            self.attention_pools = nn.ModuleList([
                AttentionPooling(hidden_dim) 
                for _ in range(num_scales)
            ])
        elif pooling_method == "gated":
            self.gated_pools = nn.ModuleList([
                GatedPooling(hidden_dim) 
                for _ in range(num_scales)
            ])
        
        # Scale combination layer
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical pooling.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch indices [num_nodes]
            pos: Node positions [num_nodes, 2]
            
        Returns:
            Dictionary containing hierarchical representations
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Generate hierarchical clusters for each scale
        hierarchical_outputs = {}
        scale_representations = []
        
        for scale_idx, radius in enumerate(self.radii):
            # Create clusters for this scale
            cluster_assignments = self._create_spatial_clusters(
                pos, batch, radius
            )
            
            # Pool features within clusters
            if self.pooling_method == "attention":
                pooled_features = self.attention_pools[scale_idx](
                    x, cluster_assignments, batch
                )
            elif self.pooling_method == "gated":
                pooled_features = self.gated_pools[scale_idx](
                    x, cluster_assignments, batch
                )
            else:
                pooled_features = self._simple_pooling(
                    x, cluster_assignments, method=self.pooling_method
                )
            
            scale_representations.append(pooled_features)
            hierarchical_outputs[f"scale_{scale_idx}"] = pooled_features
        
        # Combine representations from all scales
        combined_repr = torch.cat(scale_representations, dim=-1)
        fused_repr = self.scale_fusion(combined_repr)
        
        hierarchical_outputs["fused"] = fused_repr
        hierarchical_outputs["scales"] = scale_representations
        
        return hierarchical_outputs
    
    def _create_spatial_clusters(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        radius: float
    ) -> torch.Tensor:
        """
        Create spatial clusters based on distance threshold.
        
        Args:
            pos: Node positions [num_nodes, 2]
            batch: Batch indices [num_nodes]
            radius: Clustering radius
            
        Returns:
            Cluster assignments [num_nodes]
        """
        if pos is None:
            # Fallback to simple sequential clustering
            num_nodes = batch.size(0)
            cluster_size = max(self.min_nodes_per_cluster, num_nodes // 10)
            return torch.arange(num_nodes, device=batch.device) // cluster_size
        
        device = pos.device
        num_nodes = pos.size(0)
        cluster_assignments = torch.full(
            (num_nodes,), -1, dtype=torch.long, device=device
        )
        
        cluster_id = 0
        
        # Process each batch separately
        for batch_idx in batch.unique():
            batch_mask = batch == batch_idx
            batch_pos = pos[batch_mask]
            batch_indices = torch.where(batch_mask)[0]
            
            # Greedy clustering within this batch
            remaining_nodes = torch.ones(
                batch_pos.size(0), dtype=torch.bool, device=device
            )
            
            while remaining_nodes.sum() > 0:
                # Select a random unclustered node as cluster center
                available_indices = torch.where(remaining_nodes)[0]
                if len(available_indices) == 0:
                    break
                
                center_idx = available_indices[0]  # Take first available
                center_pos = batch_pos[center_idx]
                
                # Find all nodes within radius
                distances = torch.norm(batch_pos - center_pos, dim=1)
                within_radius = (distances <= radius) & remaining_nodes
                
                # Assign cluster ID
                cluster_nodes = batch_indices[within_radius]
                cluster_assignments[cluster_nodes] = cluster_id
                
                # Mark these nodes as processed
                remaining_nodes[within_radius] = False
                
                cluster_id += 1
        
        return cluster_assignments
    
    def _simple_pooling(
        self,
        x: torch.Tensor,
        cluster_assignments: torch.Tensor,
        method: str = "mean"
    ) -> torch.Tensor:
        """
        Simple pooling operations.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            cluster_assignments: Cluster assignments [num_nodes]
            method: Pooling method ('mean', 'max', 'sum')
            
        Returns:
            Pooled features [num_clusters, hidden_dim]
        """
        if method == "mean":
            return scatter_mean(x, cluster_assignments, dim=0)
        elif method == "max":
            return scatter_max(x, cluster_assignments, dim=0)[0]
        elif method == "sum":
            return scatter_add(x, cluster_assignments, dim=0)
        else:
            raise ValueError(f"Unknown pooling method: {method}")


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for hierarchical representations.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        # Attention components
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable query for pooling
        self.pool_query = nn.Parameter(torch.randn(1, num_heads, self.head_dim))
        
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        x: torch.Tensor,
        cluster_assignments: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention pooling within clusters.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            cluster_assignments: Cluster assignments [num_nodes]
            batch: Batch indices [num_nodes]
            
        Returns:
            Pooled features [num_nodes, hidden_dim]
        """
        # Project to keys and values
        keys = self.key_proj(x).view(-1, self.num_heads, self.head_dim)
        values = self.value_proj(x).view(-1, self.num_heads, self.head_dim)
        
        # Expand query for all nodes
        num_nodes = x.size(0)
        queries = self.pool_query.expand(num_nodes, -1, -1)
        
        # Compute attention scores
        attention_scores = torch.einsum("nhd,nhd->nh", queries, keys) * self.scale
        
        # Apply softmax within clusters
        attention_weights = torch.zeros_like(attention_scores)
        for cluster_id in cluster_assignments.unique():
            if cluster_id == -1:  # Skip unassigned nodes
                continue
            
            cluster_mask = cluster_assignments == cluster_id
            cluster_scores = attention_scores[cluster_mask]
            cluster_weights = F.softmax(cluster_scores, dim=0)
            attention_weights[cluster_mask] = cluster_weights
        
        # Apply attention to values
        attended_values = attention_weights.unsqueeze(-1) * values
        
        # Pool within clusters
        pooled_features = scatter_mean(
            attended_values.view(-1, self.hidden_dim),
            cluster_assignments,
            dim=0
        )
        
        # Project output
        output = self.out_proj(pooled_features)
        
        # Map back to original node indices
        node_outputs = output[cluster_assignments]
        
        return node_outputs


class GatedPooling(nn.Module):
    """
    Gated pooling mechanism for hierarchical representations.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Gating mechanism
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )
        
        # Feature transformation
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        cluster_assignments: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply gated pooling within clusters.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            cluster_assignments: Cluster assignments [num_nodes]
            batch: Batch indices [num_nodes]
            
        Returns:
            Pooled features [num_nodes, hidden_dim]
        """
        # Compute gates and transformed features
        gates = self.gate_proj(x)
        features = self.feature_proj(x)
        
        # Apply gating
        gated_features = gates * features
        
        # Pool within clusters
        pooled_features = scatter_mean(gated_features, cluster_assignments, dim=0)
        
        # Map back to original node indices
        node_outputs = pooled_features[cluster_assignments]
        
        return node_outputs


class AdaptiveHierarchicalPooling(nn.Module):
    """
    Adaptive hierarchical pooling that learns optimal scales.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        max_scales: int = 6,
        min_radius: float = 25.0,
        max_radius: float = 500.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_scales = max_scales
        self.min_radius = min_radius
        self.max_radius = max_radius
        
        # Learnable scale parameters
        self.scale_logits = nn.Parameter(
            torch.linspace(
                np.log(min_radius), 
                np.log(max_radius), 
                max_scales
            )
        )
        
        # Scale importance weights
        self.scale_importance = nn.Parameter(torch.ones(max_scales))
        
        # Pooling layers
        self.pooling_layers = nn.ModuleList([
            AttentionPooling(hidden_dim) 
            for _ in range(max_scales)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * max_scales, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with adaptive scale selection.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch indices [num_nodes]
            pos: Node positions [num_nodes, 2]
            
        Returns:
            Hierarchical representations [num_nodes, hidden_dim]
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Get adaptive radii
        radii = torch.exp(self.scale_logits)
        
        # Compute importance weights
        importance_weights = F.softmax(self.scale_importance, dim=0)
        
        # Process each scale
        scale_outputs = []
        for i, (radius, pooling_layer) in enumerate(
            zip(radii, self.pooling_layers)
        ):
            # Create clusters for this scale
            cluster_assignments = self._create_spatial_clusters(
                pos, batch, radius.item()
            )
            
            # Pool features
            pooled = pooling_layer(x, cluster_assignments, batch)
            
            # Weight by importance
            weighted_pooled = importance_weights[i] * pooled
            scale_outputs.append(weighted_pooled)
        
        # Fuse all scales
        concatenated = torch.cat(scale_outputs, dim=-1)
        fused = self.fusion(concatenated)
        
        return fused
    
    def _create_spatial_clusters(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        radius: float
    ) -> torch.Tensor:
        """Create spatial clusters (same as in HierarchicalPooling)."""
        if pos is None:
            num_nodes = batch.size(0)
            cluster_size = max(3, num_nodes // 10)
            return torch.arange(num_nodes, device=batch.device) // cluster_size
        
        device = pos.device
        num_nodes = pos.size(0)
        cluster_assignments = torch.full(
            (num_nodes,), -1, dtype=torch.long, device=device
        )
        
        cluster_id = 0
        
        for batch_idx in batch.unique():
            batch_mask = batch == batch_idx
            batch_pos = pos[batch_mask]
            batch_indices = torch.where(batch_mask)[0]
            
            remaining_nodes = torch.ones(
                batch_pos.size(0), dtype=torch.bool, device=device
            )
            
            while remaining_nodes.sum() > 0:
                available_indices = torch.where(remaining_nodes)[0]
                if len(available_indices) == 0:
                    break
                
                center_idx = available_indices[0]
                center_pos = batch_pos[center_idx]
                
                distances = torch.norm(batch_pos - center_pos, dim=1)
                within_radius = (distances <= radius) & remaining_nodes
                
                cluster_nodes = batch_indices[within_radius]
                cluster_assignments[cluster_nodes] = cluster_id
                
                remaining_nodes[within_radius] = False
                cluster_id += 1
        
        return cluster_assignments