"""
SpatialSpot: represents a single spatial transcriptomics measurement spot.

In Visium/Slide-seq/MERFISH data, each spot captures gene expression
from a small tissue area at a known (x, y) coordinate.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SpatialSpot:
    """
    A single spatial transcriptomics spot.

    Attributes
    ----------
    x, y : float
        Tissue-section coordinates (microns or array indices).
    expression : np.ndarray, shape (n_genes,)
        Raw UMI counts or normalized expression per gene.
    spot_id : str
        Unique identifier (barcode in Visium, cell-id in MERFISH).
    cell_type : str | None
        Annotated cell type label (optional).
    """

    x: float
    y: float
    expression: np.ndarray
    spot_id: str = ""
    cell_type: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def coords(self) -> np.ndarray:
        """Return (x, y) as a 2-D coordinate array."""
        return np.array([self.x, self.y], dtype=np.float32)

    @property
    def n_genes(self) -> int:
        return len(self.expression)

    @property
    def total_counts(self) -> float:
        return float(self.expression.sum())

    # ------------------------------------------------------------------ #
    # Normalization helpers                                                #
    # ------------------------------------------------------------------ #

    def log_normalize(self, scale_factor: float = 1e4) -> np.ndarray:
        """Log-normalize expression (like Seurat/Scanpy default)."""
        total = self.total_counts
        if total == 0:
            return np.zeros_like(self.expression, dtype=np.float32)
        normed = self.expression / total * scale_factor
        return np.log1p(normed).astype(np.float32)

    def normalized_expression(self, scale_factor: float = 1e4) -> np.ndarray:
        """Return log-normalized expression vector (does not modify in place)."""
        return self.log_normalize(scale_factor)

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        ct = f", cell_type={self.cell_type!r}" if self.cell_type else ""
        return (
            f"SpatialSpot(id={self.spot_id!r}, x={self.x:.1f}, y={self.y:.1f}, "
            f"n_genes={self.n_genes}{ct})"
        )
