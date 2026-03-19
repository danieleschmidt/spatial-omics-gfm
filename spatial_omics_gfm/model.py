"""
SpatialGraphTransformer: graph transformer for spatial transcriptomics.

Architecture
------------
1. Gene expression encoder  — linear projection from n_genes → hidden_dim
2. Spatial position encoder — sinusoidal / learnable 2D encoding → hidden_dim
3. Graph Transformer layers  (2–3 layers) using sparse attention over k-NN graph
4. Output head (configurable: domain classification, expression denoising, …)

The attention mechanism is a simplified Graph Attention Network (GAT) variant
where edge features (distance, angle) are incorporated via a learned bias term.
No torch_geometric needed — uses pure PyTorch sparse/gather ops.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =========================================================================== #
# Spatial Position Encoding                                                    #
# =========================================================================== #

class SpatialPositionEncoding(nn.Module):
    """
    Encode (x, y) coordinates into a hidden_dim vector.

    Uses sinusoidal features (like NeRF / Fourier features) concatenated with
    a learnable linear projection, giving the model both explicit frequency
    information and the flexibility to learn spatial patterns.
    """

    def __init__(self, hidden_dim: int, n_freqs: int = 8) -> None:
        super().__init__()
        self.n_freqs = n_freqs
        # Frequencies: 2^0, 2^1, …, 2^(n_freqs-1)
        freqs = 2.0 ** torch.arange(n_freqs).float()
        self.register_buffer("freqs", freqs)

        fourier_dim = 4 * n_freqs  # sin+cos × x+y
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, coords: Tensor) -> Tensor:
        """
        Parameters
        ----------
        coords : (N, 2)  — (x, y) in arbitrary units

        Returns
        -------
        Tensor : (N, hidden_dim)
        """
        # Normalise to roughly [0, 2π] per batch
        mn = coords.min(0).values
        mx = coords.max(0).values
        span = (mx - mn).clamp(min=1.0)
        c = (coords - mn) / span * 2 * math.pi  # (N, 2)

        # Sinusoidal features
        # c[:, 0] → x, c[:, 1] → y, freqs (n_freqs,)
        x_feat = c[:, 0:1] * self.freqs  # (N, n_freqs)
        y_feat = c[:, 1:2] * self.freqs  # (N, n_freqs)
        fourier = torch.cat(
            [x_feat.sin(), x_feat.cos(), y_feat.sin(), y_feat.cos()], dim=-1
        )  # (N, 4*n_freqs)
        return self.proj(fourier)


# =========================================================================== #
# Graph Attention Layer                                                        #
# =========================================================================== #

class SpatialGraphAttnLayer(nn.Module):
    """
    Single graph-attention layer with spatial edge bias.

    For each node i, attention is computed over its k-NN neighbours.
    Edge features (distance, angle) are projected to a scalar bias added
    to the attention logit before softmax.

    Returns updated node features after residual + LayerNorm.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        edge_dim: int = 2,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge bias: project (distance, angle) → num_heads scalar
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, num_heads * 2),
            nn.GELU(),
            nn.Linear(num_heads * 2, num_heads),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,          # (N, hidden_dim)
        edge_index: Tensor, # (2, E)
        edge_attr: Tensor,  # (E, edge_dim)
    ) -> Tensor:
        N = x.size(0)
        residual = x
        x = self.norm1(x)

        Q = self.q_proj(x).view(N, self.num_heads, self.head_dim)  # (N, H, D)
        K = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(N, self.num_heads, self.head_dim)

        src, dst = edge_index[0], edge_index[1]  # both (E,)

        # Attention logits for each edge (dst attends to src)
        q_dst = Q[dst]  # (E, H, D)
        k_src = K[src]  # (E, H, D)
        attn_logit = (q_dst * k_src).sum(-1) * self.scale  # (E, H)

        # Add spatial edge bias
        e_bias = self.edge_bias(edge_attr)  # (E, H)
        attn_logit = attn_logit + e_bias

        # Softmax per destination node (segment softmax)
        attn_logit_exp = attn_logit - attn_logit.max()  # numerical stability
        attn_exp = attn_logit_exp.exp()  # (E, H)

        # Aggregate denominator per dst node
        denom = torch.zeros(N, self.num_heads, device=x.device)
        denom.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.num_heads), attn_exp)
        attn_w = attn_exp / (denom[dst] + 1e-8)  # (E, H)
        attn_w = self.dropout(attn_w)

        # Weighted sum of values
        v_src = V[src]  # (E, H, D)
        weighted = attn_w.unsqueeze(-1) * v_src  # (E, H, D)
        agg = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        agg.scatter_add_(
            0,
            dst.view(-1, 1, 1).expand(-1, self.num_heads, self.head_dim),
            weighted,
        )
        agg = agg.view(N, self.hidden_dim)
        out = self.out_proj(agg)

        # Residual 1
        x = residual + self.dropout(out)

        # FFN + residual 2
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# =========================================================================== #
# SpatialGraphTransformer                                                      #
# =========================================================================== #

class SpatialGraphTransformer(nn.Module):
    """
    Graph Transformer for spatial transcriptomics data.

    Takes a SpatialGraph (or raw tensors) and produces per-node embeddings
    that capture both gene expression patterns and spatial context.

    Parameters
    ----------
    n_genes     : int  — input gene expression dimensionality
    hidden_dim  : int  — internal embedding size (default 128)
    num_layers  : int  — number of graph-attention layers (default 3)
    num_heads   : int  — attention heads per layer (default 4)
    dropout     : float
    n_classes   : int | None — if set, adds a classification head
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        n_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # Expression encoder
        self.expr_encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Spatial position encoder
        self.pos_encoder = SpatialPositionEncoding(hidden_dim)

        # Fusion gate: blend expression + position embeddings
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

        # Stack of graph-attention layers
        self.layers = nn.ModuleList([
            SpatialGraphAttnLayer(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output head (optional classification)
        if n_classes is not None:
            self.classifier = nn.Linear(hidden_dim, n_classes)
        else:
            self.classifier = None

    # ------------------------------------------------------------------ #

    def encode(
        self,
        node_features: Tensor,  # (N, n_genes)
        coords: Tensor,         # (N, 2)
        edge_index: Tensor,     # (2, E)
        edge_attr: Tensor,      # (E, 2)
    ) -> Tensor:
        """Return per-node embeddings, shape (N, hidden_dim)."""
        expr_emb = self.expr_encoder(node_features)
        pos_emb = self.pos_encoder(coords)

        # Fuse expression + spatial positional information
        x = self.fusion(torch.cat([expr_emb, pos_emb], dim=-1))

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        return x

    def forward(
        self,
        node_features: Tensor,
        coords: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Returns
        -------
        embeddings : (N, hidden_dim)
        logits     : (N, n_classes) or None
        """
        emb = self.encode(node_features, coords, edge_index, edge_attr)
        logits = self.classifier(emb) if self.classifier is not None else None
        return emb, logits

    @classmethod
    def from_graph(
        cls,
        graph,  # SpatialGraph
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        n_classes: Optional[int] = None,
    ) -> "SpatialGraphTransformer":
        """Convenience constructor that reads n_genes from a SpatialGraph."""
        return cls(
            n_genes=graph.n_genes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            n_classes=n_classes,
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"SpatialGraphTransformer("
            f"n_genes={self.n_genes}, hidden_dim={self.hidden_dim}, "
            f"layers={len(self.layers)}, params={self.parameter_count():,})"
        )
