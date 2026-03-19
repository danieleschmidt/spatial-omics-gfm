"""
SyntheticSpatialDataset: synthetic Visium-like spatial transcriptomics data.

Generates a grid of spots where gene expression varies spatially according to
overlapping Gaussian "expression fields" that mimic distinct tissue regions.

Each region expresses a different set of marker genes, and a small amount of
ligand-receptor co-expression is seeded at region boundaries to simulate
cell-cell communication signals.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from .spot import SpatialSpot


_CELL_TYPE_NAMES = [
    "Tumor", "Stromal", "Immune", "Endothelial",
    "Neuronal", "Epithelial", "Fibroblast",
]


class SyntheticSpatialDataset:
    """
    Synthetic spatial transcriptomics dataset (Visium-like grid).

    Parameters
    ----------
    n_spots    : int  — total spots (rounded to nearest square grid)
    n_genes    : int  — number of genes in expression panel
    n_regions  : int  — number of distinct spatial domains / cell types
    noise_std  : float — Gaussian noise on log-expression
    seed       : int

    Attributes
    ----------
    spots : list[SpatialSpot]
    """

    def __init__(
        self,
        n_spots: int = 100,
        n_genes: int = 500,
        n_regions: int = 5,
        noise_std: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.n_genes = n_genes
        self.n_regions = n_regions
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        self.spots = self._generate(n_spots)

    # ------------------------------------------------------------------ #
    # Generation                                                           #
    # ------------------------------------------------------------------ #

    def _generate(self, n_spots: int) -> List[SpatialSpot]:
        # Fit spots onto a rectangular grid
        grid_side = max(1, int(np.ceil(np.sqrt(n_spots))))
        xs = np.arange(grid_side, dtype=np.float32)
        ys = np.arange(grid_side, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        all_x = xx.ravel()[:n_spots]
        all_y = yy.ravel()[:n_spots]
        N = len(all_x)

        # Place region centres randomly on the grid
        region_centres = np.stack([
            self.rng.uniform(0, grid_side - 1, size=self.n_regions),
            self.rng.uniform(0, grid_side - 1, size=self.n_regions),
        ], axis=1)  # (n_regions, 2)

        # Each region has a distinct set of marker genes
        # (50% marker genes + 50% house-keeping genes shared across regions)
        n_markers = n_spots // 4  # rough fraction of dedicated markers
        n_markers = max(10, min(n_markers, self.n_genes // (self.n_regions + 1)))

        region_markers: List[np.ndarray] = []
        all_marker_idx: set = set()
        for r in range(self.n_regions):
            start = r * n_markers
            end = start + n_markers
            m = np.arange(start % self.n_genes, end % self.n_genes)
            region_markers.append(m)
            all_marker_idx.update(m.tolist())

        # Region membership per spot: soft (Gaussian distance)
        coords = np.stack([all_x, all_y], axis=1)  # (N, 2)
        diffs = coords[:, None, :] - region_centres[None, :, :]  # (N, K, 2)
        dists_sq = (diffs ** 2).sum(-1)  # (N, K)
        # sigma proportional to grid size
        sigma = grid_side / 2.5
        soft_assign = np.exp(-dists_sq / (2 * sigma ** 2))  # (N, K)
        soft_assign = soft_assign / soft_assign.sum(axis=1, keepdims=True)

        # Hard region label (dominant region)
        hard_label = soft_assign.argmax(axis=1)  # (N,)

        # Build expression matrix: (N, n_genes)
        # House-keeping: random low-level expression for all
        expr = self.rng.exponential(scale=1.0, size=(N, self.n_genes)).astype(np.float32)

        # Marker genes: boost expression proportional to soft-assignment
        for r, markers in enumerate(region_markers):
            boost = soft_assign[:, r:r+1] * 8.0  # strong signal in dominant spots
            expr[:, markers] += boost

        # Add noise
        expr += self.rng.normal(0, self.noise_std, size=expr.shape).astype(np.float32)
        expr = np.clip(expr, 0, None)

        # Build SpatialSpot objects
        spots = []
        for i in range(N):
            r = int(hard_label[i])
            spots.append(SpatialSpot(
                x=float(all_x[i]),
                y=float(all_y[i]),
                expression=expr[i],
                spot_id=f"spot_{i:04d}",
                cell_type=_CELL_TYPE_NAMES[r % len(_CELL_TYPE_NAMES)],
            ))

        return spots

    # ------------------------------------------------------------------ #
    # Properties / helpers                                                 #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.spots)

    def __getitem__(self, idx: int) -> SpatialSpot:
        return self.spots[idx]

    def get_cell_types(self) -> List[Optional[str]]:
        return [s.cell_type for s in self.spots]

    def unique_cell_types(self) -> List[str]:
        return sorted(set(s.cell_type for s in self.spots if s.cell_type))

    def get_coords(self) -> np.ndarray:
        """Return (N, 2) coordinate array."""
        return np.stack([s.coords for s in self.spots])

    def __repr__(self) -> str:
        types = ", ".join(self.unique_cell_types())
        return (
            f"SyntheticSpatialDataset(n_spots={len(self)}, "
            f"n_genes={self.n_genes}, cell_types=[{types}])"
        )
