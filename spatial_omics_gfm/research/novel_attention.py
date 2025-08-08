"""
Novel spatial attention mechanisms for enhanced spatial transcriptomics analysis.
Implements cutting-edge attention architectures optimized for spatial data.
"""

import logging
import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_self_loops
import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveSpatialAttention(MessagePassing):
    """
    Adaptive spatial attention that learns optimal attention patterns
    based on local tissue architecture and cellular density.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        max_distance: float = 500.0,
        adaptive_radius: bool = True,
        temperature_learning: bool = True,
        dropout: float = 0.1
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_distance = max_distance
        self.adaptive_radius = adaptive_radius
        self.temperature_learning = temperature_learning
        self.dropout = dropout
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Spatial encoding
        self.spatial_encoder = SpatialRelativePositionEncoder(
            hidden_dim=self.head_dim,
            max_distance=max_distance
        )
        
        # Adaptive radius learning
        if adaptive_radius:
            self.radius_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # Temperature learning for attention sharpness
        if temperature_learning:
            self.temperature = nn.Parameter(torch.ones(num_heads))
        else:
            self.register_buffer('temperature', torch.ones(num_heads))
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Density-aware weighting
        self.density_encoder = LocalDensityEncoder(hidden_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.temperature_learning:
            nn.init.constant_(self.temperature, 1.0)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        spatial_coords: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with adaptive spatial attention.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            spatial_coords: Spatial coordinates [num_nodes, 2]
            batch: Batch indices [num_nodes]
            
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        num_nodes = x.size(0)
        
        # Project to Q, K, V
        q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Compute density-aware features
        density_features = self.density_encoder(x, edge_index, spatial_coords)
        
        # Propagate messages
        out = self.propagate(
            edge_index, q=q, k=k, v=v,
            edge_attr=edge_attr,
            spatial_coords=spatial_coords,
            density_features=density_features,
            size=(num_nodes, num_nodes)
        )
        
        # Reshape and project output
        out = out.view(num_nodes, self.hidden_dim)
        out = self.out_proj(out)
        out = self.dropout_layer(out)
        
        return out
    
    def message(
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        spatial_coords_i: Optional[torch.Tensor],
        spatial_coords_j: Optional[torch.Tensor],
        density_features_i: torch.Tensor,
        density_features_j: torch.Tensor,
        index: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention messages."""
        # Basic attention computation
        attention_scores = (q_i * k_j).sum(dim=-1) / math.sqrt(self.head_dim)
        
        # Apply temperature scaling per head
        attention_scores = attention_scores / self.temperature.view(1, -1)
        
        # Add spatial relative position encoding
        if spatial_coords_i is not None and spatial_coords_j is not None:
            spatial_encoding = self.spatial_encoder(
                spatial_coords_i, spatial_coords_j
            )
            attention_scores = attention_scores + spatial_encoding
        
        # Add density-aware modulation
        density_similarity = F.cosine_similarity(
            density_features_i, density_features_j, dim=-1
        ).unsqueeze(-1)
        attention_scores = attention_scores * (1 + 0.5 * density_similarity)
        
        # Apply edge features if available
        if edge_attr is not None:
            edge_weights = edge_attr[:, 0].unsqueeze(-1).unsqueeze(-1)
            attention_scores = attention_scores * edge_weights
        
        # Softmax over neighbors
        attention_probs = softmax(attention_scores, index)
        attention_probs = self.dropout_layer(attention_probs)
        
        # Apply attention to values
        messages = attention_probs.unsqueeze(-1) * v_j
        
        return messages


class SpatialRelativePositionEncoder(nn.Module):
    """
    Encodes relative spatial positions with learnable distance-based embeddings.
    """
    
    def __init__(self, hidden_dim: int, max_distance: float = 500.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_distance = max_distance
        
        # Distance embedding
        self.distance_embedding = nn.Embedding(100, hidden_dim // 2)
        
        # Direction embedding (angle-based)
        self.direction_embedding = nn.Embedding(16, hidden_dim // 2)
        
        # Combine spatial features
        self.spatial_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        coords_i: torch.Tensor,
        coords_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode relative spatial positions.
        
        Args:
            coords_i: Source coordinates [num_edges, 2]
            coords_j: Target coordinates [num_edges, 2]
            
        Returns:
            Spatial encoding [num_edges, num_heads]
        """
        # Compute relative positions
        rel_pos = coords_j - coords_i  # [num_edges, 2]
        
        # Compute distances
        distances = torch.norm(rel_pos, dim=-1)  # [num_edges]
        
        # Compute angles
        angles = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])  # [num_edges]
        
        # Discretize distances for embedding
        distance_bins = torch.clamp(
            (distances / self.max_distance * 99).long(),
            0, 99
        )
        
        # Discretize angles for embedding
        angle_bins = torch.clamp(
            ((angles + math.pi) / (2 * math.pi) * 15).long(),
            0, 15
        )
        
        # Get embeddings
        distance_emb = self.distance_embedding(distance_bins)  # [num_edges, hidden_dim//2]
        direction_emb = self.direction_embedding(angle_bins)   # [num_edges, hidden_dim//2]
        
        # Combine embeddings
        spatial_features = torch.cat([distance_emb, direction_emb], dim=-1)
        
        # Project to scalar per head
        spatial_encoding = self.spatial_mlp(spatial_features).squeeze(-1)
        
        return spatial_encoding


class LocalDensityEncoder(nn.Module):
    """
    Encodes local cellular density patterns for density-aware attention.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.density_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),  # density + local stats
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        spatial_coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute local density features.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            spatial_coords: Spatial coordinates [num_nodes, 2]
            
        Returns:
            Density features [num_nodes, hidden_dim]
        """
        num_nodes = x.size(0)
        
        # Compute node degrees (connectivity density)
        row, col = edge_index
        degree = torch.zeros(num_nodes, dtype=torch.float, device=x.device)
        degree.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        
        # Compute local spatial density if coordinates available
        if spatial_coords is not None:
            spatial_density = self._compute_spatial_density(
                spatial_coords, edge_index
            )
        else:
            spatial_density = torch.ones_like(degree)
        
        # Compute local feature variance
        feature_variance = self._compute_local_feature_variance(x, edge_index)
        
        # Combine density metrics
        density_features = torch.stack([
            degree / (degree.max() + 1e-8),
            spatial_density,
            feature_variance
        ], dim=-1)
        
        # Project to feature space
        encoded_density = self.density_mlp(density_features)
        
        return encoded_density
    
    def _compute_spatial_density(
        self,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        radius: float = 50.0
    ) -> torch.Tensor:
        """Compute spatial density within radius."""
        num_nodes = coords.size(0)
        density = torch.zeros(num_nodes, device=coords.device)
        
        # Compute pairwise distances
        distances = torch.cdist(coords, coords)
        
        # Count neighbors within radius
        for i in range(num_nodes):
            neighbors_in_radius = (distances[i] <= radius).sum().float() - 1  # Exclude self
            density[i] = neighbors_in_radius
        
        # Normalize
        density = density / (density.max() + 1e-8)
        
        return density
    
    def _compute_local_feature_variance(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute local feature variance."""
        num_nodes = x.size(0)
        row, col = edge_index
        
        # Aggregate neighbor features
        neighbor_features = torch.zeros_like(x)
        neighbor_counts = torch.zeros(num_nodes, dtype=torch.float, device=x.device)
        
        neighbor_features.scatter_add_(0, row.unsqueeze(1).expand(-1, x.size(1)), x[col])
        neighbor_counts.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        
        # Compute mean neighbor features
        neighbor_counts = neighbor_counts.clamp(min=1).unsqueeze(1)
        neighbor_mean = neighbor_features / neighbor_counts
        
        # Compute variance
        feature_variance = torch.norm(x - neighbor_mean, dim=-1)
        feature_variance = feature_variance / (feature_variance.max() + 1e-8)
        
        return feature_variance


class HierarchicalSpatialAttention(nn.Module):
    """
    Hierarchical attention that operates at multiple spatial scales
    to capture both local and global tissue organization patterns.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_scales: int = 3,
        scale_factors: Optional[list] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_scales = num_scales
        self.dropout = dropout
        
        if scale_factors is None:
            scale_factors = [1.0, 2.0, 4.0][:num_scales]
        self.scale_factors = scale_factors
        
        # Multi-scale attention layers
        self.scale_attentions = nn.ModuleList([
            AdaptiveSpatialAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                max_distance=500.0 * scale,
                dropout=dropout
            )
            for scale in scale_factors
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Gating mechanism for scale importance
        self.scale_gates = nn.Sequential(
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        spatial_coords: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with hierarchical spatial attention.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            spatial_coords: Spatial coordinates [num_nodes, 2]
            batch: Batch indices [num_nodes]
            
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        scale_outputs = []
        
        # Apply attention at each scale
        for i, attention_layer in enumerate(self.scale_attentions):
            # Modify edge attributes for this scale
            if edge_attr is not None:
                scale_edge_attr = edge_attr / self.scale_factors[i]
            else:
                scale_edge_attr = None
            
            scale_output = attention_layer(
                x, edge_index, scale_edge_attr, spatial_coords, batch
            )
            scale_outputs.append(scale_output)
        
        # Concatenate scale outputs
        multi_scale_features = torch.cat(scale_outputs, dim=-1)
        
        # Fuse scales
        fused_features = self.scale_fusion(multi_scale_features)
        
        # Compute scale importance gates
        scale_importance = self.scale_gates(x)
        
        # Weight and combine with original features
        weighted_scales = sum(
            importance[:, i:i+1] * scale_outputs[i]
            for i, importance in enumerate(scale_importance.chunk(self.num_scales, dim=1))
        )
        
        # Residual connection
        output = x + fused_features + weighted_scales
        
        return output


class ContextualSpatialAttention(nn.Module):
    """
    Context-aware spatial attention that adapts based on local tissue context
    and biological markers to focus on relevant spatial relationships.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        context_dim: int = 64,
        num_contexts: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.context_dim = context_dim
        self.num_contexts = num_contexts
        
        # Context identification network
        self.context_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_contexts),
            nn.Softmax(dim=-1)
        )
        
        # Context-specific attention parameters
        self.context_attention_weights = nn.Parameter(
            torch.randn(num_contexts, num_heads, hidden_dim, hidden_dim)
        )
        
        # Base attention mechanism
        self.base_attention = AdaptiveSpatialAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.context_attention_weights)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        spatial_coords: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with contextual spatial attention.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            spatial_coords: Spatial coordinates [num_nodes, 2]
            batch: Batch indices [num_nodes]
            
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        num_nodes = x.size(0)
        
        # Identify tissue contexts
        context_probs = self.context_classifier(x)  # [num_nodes, num_contexts]
        
        # Apply base attention
        attended_features = self.base_attention(
            x, edge_index, edge_attr, spatial_coords, batch
        )
        
        # Apply context-specific transformations
        context_features = []
        for i in range(self.num_contexts):
            # Get context-specific weights
            context_weight = self.context_attention_weights[i]  # [num_heads, hidden_dim, hidden_dim]
            
            # Apply context transformation
            context_transformed = torch.einsum(
                'nhd,hdi->nhi', 
                attended_features.view(num_nodes, self.num_heads, -1),
                context_weight
            ).view(num_nodes, -1)
            
            # Weight by context probability
            weighted_context = context_probs[:, i:i+1] * context_transformed
            context_features.append(weighted_context)
        
        # Sum context-weighted features
        final_context_features = sum(context_features)
        
        # Create context embedding
        context_embedding = torch.matmul(
            context_probs, 
            torch.randn(self.num_contexts, self.context_dim, device=x.device)
        )
        
        # Fuse with context information
        combined_features = torch.cat([final_context_features, context_embedding], dim=-1)
        output = self.context_fusion(combined_features)
        
        return output


def create_novel_attention_layer(
    attention_type: str,
    hidden_dim: int,
    num_heads: int = 8,
    **kwargs
) -> nn.Module:
    """
    Factory function to create novel attention layers.
    
    Args:
        attention_type: Type of attention ('adaptive', 'hierarchical', 'contextual')
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        **kwargs: Additional arguments
        
    Returns:
        Attention layer instance
    """
    if attention_type == 'adaptive':
        return AdaptiveSpatialAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            **kwargs
        )
    elif attention_type == 'hierarchical':
        return HierarchicalSpatialAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            **kwargs
        )
    elif attention_type == 'contextual':
        return ContextualSpatialAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


class NovelAttentionBenchmark:
    """
    Benchmark suite for novel attention mechanisms.
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        
    def benchmark_attention_mechanisms(
        self,
        hidden_dim: int = 256,
        num_nodes: int = 1000,
        num_edges: int = 5000,
        num_heads: int = 8
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark different attention mechanisms.
        
        Returns:
            Performance metrics for each attention type
        """
        logger.info("Benchmarking novel attention mechanisms...")
        
        # Create test data
        x = torch.randn(num_nodes, hidden_dim, device=self.device)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=self.device)
        edge_attr = torch.randn(num_edges, 1, device=self.device)
        spatial_coords = torch.randn(num_nodes, 2, device=self.device)
        
        attention_types = ['adaptive', 'hierarchical', 'contextual']
        results = {}
        
        for attention_type in attention_types:
            logger.info(f"Benchmarking {attention_type} attention...")
            
            # Create attention layer
            attention_layer = create_novel_attention_layer(
                attention_type=attention_type,
                hidden_dim=hidden_dim,
                num_heads=num_heads
            ).to(self.device)
            
            # Benchmark forward pass
            import time
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = attention_layer(x, edge_index, edge_attr, spatial_coords)
            
            # Actual timing
            start_time = time.time()
            num_runs = 10
            
            for _ in range(num_runs):
                with torch.no_grad():
                    output = attention_layer(x, edge_index, edge_attr, spatial_coords)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs
            
            # Memory usage
            if torch.cuda.is_available() and self.device.type == 'cuda':
                memory_usage = torch.cuda.memory_allocated() / 1024**2  # MB
            else:
                memory_usage = 0
            
            # Parameter count
            param_count = sum(p.numel() for p in attention_layer.parameters())
            
            results[attention_type] = {
                'avg_forward_time_ms': avg_time * 1000,
                'memory_usage_mb': memory_usage,
                'parameter_count': param_count,
                'output_shape': list(output.shape)
            }
        
        logger.info("Attention benchmarking completed")
        return results