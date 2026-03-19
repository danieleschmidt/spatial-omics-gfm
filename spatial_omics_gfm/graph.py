"""
SpatialGraph: build a spatial k-NN graph from SpatialSpot instances.

Nodes  = spots
Edges  = spatial proximity (k nearest neighbours by Euclidean distance)
Node features = log-normalized gene expression vectors
Edge features = distance + direction between connected spots
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch import Tensor

from .spot import SpatialSpot


class SpatialGraph:
    """
    Spatial k-NN graph built from a collection of SpatialSpot objects.

    Parameters
    ----------
    spots : list of SpatialSpot
    k : int
        Number of nearest neighbours per spot (spatial proximity edges).
    scale_factor : float
        Passed to SpatialSpot.log_normalize for expression normalisation.

    Attributes (populated after construction)
    -----------------------------------------
    node_features : Tensor, shape (N, n_genes)
    coords        : Tensor, shape (N, 2)
    edge_index    : Tensor, shape (2, E)  — COO format, undirected
    edge_attr     : Tensor, shape (E, 2)  — [distance, angle]
    spot_ids      : list[str]
    cell_types    : list[str | None]
    """

    def __init__(
        self,
        spots: List[SpatialSpot],
        k: int = 6,
        scale_factor: float = 1e4,
    ) -> None:
        if len(spots) == 0:
            raise ValueError("spots list is empty")

        self.spots = spots
        self.k = k
        self.scale_factor = scale_factor

        self.spot_ids: List[str] = [s.spot_id for s in spots]
        self.cell_types: List[Optional[str]] = [s.cell_type for s in spots]

        # Build graph tensors
        self.coords, self.node_features = self._build_node_tensors()
        self.edge_index, self.edge_attr = self._build_edges()

    # ------------------------------------------------------------------ #
    # Build helpers                                                        #
    # ------------------------------------------------------------------ #

    def _build_node_tensors(self) -> Tuple[Tensor, Tensor]:
        coords_np = np.stack([s.coords for s in self.spots])  # (N, 2)
        feats_np = np.stack(
            [s.log_normalize(self.scale_factor) for s in self.spots]
        )  # (N, n_genes)
        coords = torch.from_numpy(coords_np.astype(np.float32))
        feats = torch.from_numpy(feats_np.astype(np.float32))
        return coords, feats

    def _build_edges(self) -> Tuple[Tensor, Tensor]:
        coords_np = self.coords.numpy()
        N = len(self.spots)
        k = min(self.k, N - 1)

        # Pairwise distances
        diff = coords_np[:, None, :] - coords_np[None, :, :]  # (N, N, 2)
        dists = np.linalg.norm(diff, axis=-1)                  # (N, N)

        src_list, dst_list, dist_list, angle_list = [], [], [], []
        for i in range(N):
            row = dists[i].copy()
            row[i] = np.inf                     # exclude self
            nn_idx = np.argpartition(row, k)[:k]
            for j in nn_idx:
                src_list.append(i)
                dst_list.append(j)
                dist_list.append(dists[i, j])
                dy, dx = diff[i, j, 1], diff[i, j, 0]
                angle_list.append(np.arctan2(dy, dx))

        src = torch.tensor(src_list, dtype=torch.long)
        dst = torch.tensor(dst_list, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)  # (2, E)

        dist_t = torch.tensor(dist_list, dtype=torch.float32).unsqueeze(1)
        angle_t = torch.tensor(angle_list, dtype=torch.float32).unsqueeze(1)
        edge_attr = torch.cat([dist_t, angle_t], dim=1)  # (E, 2)

        return edge_index, edge_attr

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def num_nodes(self) -> int:
        return len(self.spots)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def n_genes(self) -> int:
        return self.node_features.shape[1]

    def adjacency_list(self) -> List[List[int]]:
        """Return adjacency as list-of-lists (node → list of neighbours)."""
        adj: List[List[int]] = [[] for _ in range(self.num_nodes)]
        ei = self.edge_index.numpy()
        for s, d in zip(ei[0], ei[1]):
            adj[s].append(int(d))
        return adj

    def __repr__(self) -> str:
        return (
            f"SpatialGraph(nodes={self.num_nodes}, edges={self.num_edges}, "
            f"n_genes={self.n_genes}, k={self.k})"
        )
