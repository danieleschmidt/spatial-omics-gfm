"""
Spatial Graph Transformer: Core architecture for spatial transcriptomics analysis.

This module implements the billion-parameter Graph Foundation Model that treats
tissue sections as graphs where cells are nodes and spatial proximity defines edges.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import warnings

from .spatial_attention import SpatialAttention
from .hierarchical_pooling import HierarchicalPooling


@dataclass
class TransformerConfig:
    """Configuration for SpatialGraphTransformer."""
    num_genes: int
    hidden_dim: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    dropout: float = 0.1
    spatial_encoding_dim: int = 64
    max_neighbors: int = 10
    max_distance: int = 500
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    use_edge_features: bool = True
    use_hierarchical_pooling: bool = True


class SpatialPositionEncoding(nn.Module):
    """
    Spatial position encoding for 2D/3D coordinates.
    
    Encodes spatial coordinates using learnable embeddings combined
    with sinusoidal positional encoding.
    """
    
    def __init__(
        self, 
        spatial_dim: int = 2,
        encoding_dim: int = 64,
        max_position: int = 10000
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.encoding_dim = encoding_dim
        self.max_position = max_position
        
        # Learnable spatial embedding
        self.spatial_embed = nn.Linear(spatial_dim, encoding_dim // 2)
        
        # Sinusoidal position encoding
        self.register_buffer(
            "div_term",
            torch.exp(torch.arange(0, encoding_dim // 2, 2).float() * 
                     (-math.log(max_position) / (encoding_dim // 2)))
        )
        
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Encode spatial coordinates.
        
        Args:
            coordinates: Spatial coordinates [batch_size, spatial_dim]
            
        Returns:
            Encoded spatial features [batch_size, encoding_dim]
        """
        batch_size = coordinates.size(0)
        
        # Learnable encoding
        learned_encoding = self.spatial_embed(coordinates)
        
        # Sinusoidal encoding
        pe = torch.zeros(batch_size, self.encoding_dim // 2, device=coordinates.device)
        
        # Use first coordinate (typically x) for sinusoidal encoding
        pos = coordinates[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        pe[:, 0::2] = torch.sin(pos * self.div_term[::2])
        pe[:, 1::2] = torch.cos(pos * self.div_term[::2])
        
        # Combine learned and sinusoidal encodings
        return torch.cat([learned_encoding, pe], dim=1)


class SpatialTransformerLayer(nn.Module):
    """
    Individual Spatial Transformer layer with spatial-aware attention.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        use_edge_features: bool = True,
        max_distance: int = 500
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Spatial attention mechanism
        self.attention = SpatialAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_distance=max_distance,
            dropout=dropout,
            use_edge_features=use_edge_features
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through spatial transformer layer.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch indices [num_nodes]
            
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        # Self-attention with residual connection
        attention_out = self.attention(x, edge_index, edge_attr, batch)
        x = self.norm1(x + self.dropout(attention_out))
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class SpatialGraphTransformer(nn.Module):
    """
    Billion-parameter Graph Foundation Model for Spatial Transcriptomics.
    
    This model treats tissue sections as graphs where cells/spots are nodes
    and spatial proximity defines edges. It uses a transformer architecture
    with spatial-aware attention to model complex tissue organization patterns.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Gene expression encoder
        self.gene_encoder = nn.Sequential(
            nn.Linear(config.num_genes, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
        # Spatial position encoder
        self.spatial_encoder = SpatialPositionEncoding(
            spatial_dim=2,  # Assuming 2D coordinates
            encoding_dim=config.spatial_encoding_dim
        )
        
        # Project spatial encoding to hidden dimension
        self.spatial_proj = nn.Linear(
            config.spatial_encoding_dim, 
            config.hidden_dim
        )
        
        # Graph transformer layers
        self.layers = nn.ModuleList([
            SpatialTransformerLayer(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                activation=config.activation,
                layer_norm_eps=config.layer_norm_eps,
                use_edge_features=config.use_edge_features,
                max_distance=config.max_distance
            ) for _ in range(config.num_layers)
        ])
        
        # Hierarchical pooling for multi-scale analysis
        if config.use_hierarchical_pooling:
            self.hierarchical_pooling = HierarchicalPooling(
                hidden_dim=config.hidden_dim,
                num_scales=4
            )
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(
        self,
        gene_expression: torch.Tensor,
        spatial_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the spatial graph transformer.
        
        Args:
            gene_expression: Gene expression matrix [num_nodes, num_genes]
            spatial_coords: Spatial coordinates [num_nodes, 2]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch indices [num_nodes]
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary containing:
                - embeddings: Final node embeddings [num_nodes, hidden_dim]
                - hierarchical_embeddings: Multi-scale embeddings (if enabled)
        """
        num_nodes = gene_expression.size(0)
        
        # Encode gene expression
        gene_embeddings = self.gene_encoder(gene_expression)
        
        # Encode spatial coordinates
        spatial_embeddings = self.spatial_encoder(spatial_coords)
        spatial_embeddings = self.spatial_proj(spatial_embeddings)
        
        # Combine gene and spatial information
        x = gene_embeddings + spatial_embeddings
        
        # Store intermediate embeddings if requested
        layer_embeddings = [] if return_embeddings else None
        
        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr, batch)
            
            if return_embeddings:
                layer_embeddings.append(x.clone())
        
        # Final normalization
        x = self.final_norm(x)
        
        # Prepare output dictionary
        output = {"embeddings": x}
        
        if return_embeddings:
            output["layer_embeddings"] = layer_embeddings
        
        # Hierarchical pooling for multi-scale analysis
        if hasattr(self, "hierarchical_pooling"):
            hierarchical_emb = self.hierarchical_pooling(
                x, edge_index, batch
            )
            output["hierarchical_embeddings"] = hierarchical_emb
        
        return output
    
    def encode(
        self,
        gene_expression: torch.Tensor,
        spatial_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode inputs to get embeddings.
        
        Args:
            gene_expression: Gene expression matrix
            spatial_coords: Spatial coordinates
            edge_index: Edge connectivity
            edge_attr: Edge features
            batch: Batch indices
            
        Returns:
            Final embeddings [num_nodes, hidden_dim]
        """
        output = self.forward(
            gene_expression=gene_expression,
            spatial_coords=spatial_coords,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            return_embeddings=False
        )
        return output["embeddings"]
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: Optional[str] = None,
        **kwargs
    ) -> "SpatialGraphTransformer":
        """
        Load a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained model
            device: Device to load the model on
            **kwargs: Additional arguments
            
        Returns:
            Pre-trained SpatialGraphTransformer model
        """
        from .pretrained_models import load_pretrained_model
        return load_pretrained_model(model_name, device=device, **kwargs)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        Get parameter count breakdown.
        
        Returns:
            Dictionary with parameter counts for each component
        """
        param_counts = {}
        
        # Count parameters for each component
        param_counts["gene_encoder"] = sum(
            p.numel() for p in self.gene_encoder.parameters()
        )
        param_counts["spatial_encoder"] = sum(
            p.numel() for p in self.spatial_encoder.parameters()
        )
        param_counts["spatial_proj"] = sum(
            p.numel() for p in self.spatial_proj.parameters()
        )
        param_counts["transformer_layers"] = sum(
            p.numel() for p in self.layers.parameters()
        )
        
        if hasattr(self, "hierarchical_pooling"):
            param_counts["hierarchical_pooling"] = sum(
                p.numel() for p in self.hierarchical_pooling.parameters()
            )
        
        param_counts["total"] = sum(param_counts.values())
        
        return param_counts
    
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to save memory during training."""
        for layer in self.layers:
            layer.gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        for layer in self.layers:
            layer.gradient_checkpointing = False


def create_model_config(
    num_genes: int,
    model_size: str = "base"
) -> TransformerConfig:
    """
    Create model configuration for different model sizes.
    
    Args:
        num_genes: Number of genes in the dataset
        model_size: Model size ('base', 'large', 'xlarge')
        
    Returns:
        TransformerConfig object
    """
    size_configs = {
        "base": {
            "hidden_dim": 1024,
            "num_layers": 12,
            "num_heads": 16,
        },
        "large": {
            "hidden_dim": 1536,
            "num_layers": 24,
            "num_heads": 24,
        },
        "xlarge": {
            "hidden_dim": 2048,
            "num_layers": 36,
            "num_heads": 32,
        }
    }
    
    if model_size not in size_configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    config_dict = size_configs[model_size]
    config_dict["num_genes"] = num_genes
    
    return TransformerConfig(**config_dict)