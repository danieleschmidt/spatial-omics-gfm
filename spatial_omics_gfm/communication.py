"""
CellCommunicationInferer: infer cell-cell communication from spatial data.

Approach
--------
Ligand-receptor (L-R) co-expression across spatially adjacent spot pairs is
used to score putative signalling interactions.

For each L-R pair and each directed edge (sender → receiver):
  score = ligand_expr(sender) * receptor_expr(receiver) * proximity_weight

where proximity_weight decays exponentially with spot distance.

We also compute the Pearson correlation of expression profiles between
neighbouring spot pairs to identify spots with highly similar transcriptomes
(candidate cell-cell coupling or niche organisation).
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class LRPair:
    """A ligand-receptor gene pair."""
    ligand_idx: int    # gene index of the ligand
    receptor_idx: int  # gene index of the receptor
    name: str = ""

    def __repr__(self) -> str:
        return f"LRPair({self.name or f'L{self.ligand_idx}:R{self.receptor_idx}'})"


@dataclass
class CommunicationResult:
    """Result of a cell-communication inference run."""
    lr_pair: LRPair
    sender_idx: np.ndarray    # (M,) spot indices acting as senders
    receiver_idx: np.ndarray  # (M,) spot indices acting as receivers
    scores: np.ndarray        # (M,) interaction score per directed edge
    mean_score: float

    def top_interactions(self, n: int = 10) -> List[Tuple[int, int, float]]:
        """Return top-n (sender, receiver, score) tuples."""
        order = np.argsort(self.scores)[::-1][:n]
        return [
            (int(self.sender_idx[i]), int(self.receiver_idx[i]), float(self.scores[i]))
            for i in order
        ]

    def __repr__(self) -> str:
        return (
            f"CommunicationResult({self.lr_pair}, "
            f"mean_score={self.mean_score:.4f}, n_edges={len(self.scores)})"
        )


class CellCommunicationInferer:
    """
    Infer cell-cell communication patterns in spatial transcriptomics data.

    Parameters
    ----------
    lr_pairs : list of LRPair
        Ligand-receptor gene pairs to test.
    distance_scale : float
        Half-decay distance for the proximity weight (default 100 units).
        Interactions decay as exp(-distance / distance_scale).
    min_expr_threshold : float
        Minimum normalised expression to count a gene as "expressed".
    """

    def __init__(
        self,
        lr_pairs: List[LRPair],
        distance_scale: float = 100.0,
        min_expr_threshold: float = 0.1,
    ) -> None:
        self.lr_pairs = lr_pairs
        self.distance_scale = distance_scale
        self.min_expr_threshold = min_expr_threshold

    # ------------------------------------------------------------------ #
    # Main inference                                                       #
    # ------------------------------------------------------------------ #

    def infer(
        self,
        node_features: Tensor,  # (N, n_genes) — log-normalised expression
        edge_index: Tensor,     # (2, E) — directed k-NN edges
        edge_attr: Tensor,      # (E, 2) — [distance, angle]
    ) -> List[CommunicationResult]:
        """
        Run L-R interaction scoring over all edges.

        Returns one CommunicationResult per L-R pair.
        """
        feats = node_features.numpy() if isinstance(node_features, Tensor) else node_features
        src_arr = edge_index[0].numpy()
        dst_arr = edge_index[1].numpy()
        dists = edge_attr[:, 0].numpy()

        # Proximity weights
        prox = np.exp(-dists / self.distance_scale)

        results = []
        for lr in self.lr_pairs:
            lig = feats[:, lr.ligand_idx]    # (N,)
            rec = feats[:, lr.receptor_idx]  # (N,)

            lig_expr = np.maximum(lig[src_arr] - self.min_expr_threshold, 0)
            rec_expr = np.maximum(rec[dst_arr] - self.min_expr_threshold, 0)

            scores = lig_expr * rec_expr * prox
            mean_score = float(scores.mean())

            results.append(CommunicationResult(
                lr_pair=lr,
                sender_idx=src_arr,
                receiver_idx=dst_arr,
                scores=scores,
                mean_score=mean_score,
            ))

        return results

    # ------------------------------------------------------------------ #
    # Expression correlation (niche similarity)                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def pairwise_correlation(
        node_features: Tensor,
        edge_index: Tensor,
        top_genes: int = 200,
    ) -> np.ndarray:
        """
        Pearson correlation of expression profiles for each edge's spot pair.

        Only the top `top_genes` highest-variance genes are used, making this
        efficient even for large gene panels.

        Returns
        -------
        corr : np.ndarray, shape (E,)
            Pearson r for each (src, dst) pair.
        """
        feats = node_features.numpy() if isinstance(node_features, Tensor) else node_features
        src_arr = edge_index[0].numpy()
        dst_arr = edge_index[1].numpy()

        # Select high-variance genes
        gene_var = feats.var(axis=0)
        top_genes = min(top_genes, feats.shape[1])
        top_idx = np.argpartition(gene_var, -top_genes)[-top_genes:]
        f = feats[:, top_idx]

        # Mean-center
        f = f - f.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
        f_normed = f / norms

        # Dot product = Pearson r (already mean-centred and unit-norm)
        corr = (f_normed[src_arr] * f_normed[dst_arr]).sum(axis=1)
        return corr

    # ------------------------------------------------------------------ #
    # Spatial domain detection (spectral clustering on expression sim.)   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_spatial_domains(
        node_features: Tensor,
        coords: Tensor,
        n_domains: int = 5,
        top_genes: int = 200,
    ) -> np.ndarray:
        """
        Simple spectral clustering to assign each spot to a spatial domain.

        Uses a Gaussian kernel on expression distance (not spatial distance)
        so domains reflect transcriptional similarity with spatial continuity
        implicit via the input data.

        Returns
        -------
        labels : np.ndarray, shape (N,) — domain index per spot, 0-indexed
        """
        from sklearn.cluster import KMeans  # lightweight, in conda base

        feats = node_features.numpy() if isinstance(node_features, Tensor) else node_features

        # PCA-like: top components via SVD on high-variance genes
        gene_var = feats.var(axis=0)
        top_genes = min(top_genes, feats.shape[1])
        top_idx = np.argpartition(gene_var, -top_genes)[-top_genes:]
        f = feats[:, top_idx]

        # Mean-centre
        f = f - f.mean(axis=0)

        # Thin SVD for fast PCA (keep 20 PCs)
        n_pcs = min(20, f.shape[1], f.shape[0])
        _, _, Vt = np.linalg.svd(f, full_matrices=False)
        pcs = f @ Vt[:n_pcs].T  # (N, n_pcs)

        # Concatenate spatial coordinates (scaled)
        c = coords.numpy() if isinstance(coords, Tensor) else coords
        c_scaled = (c - c.mean(axis=0)) / (c.std(axis=0) + 1e-8) * 0.5
        X = np.concatenate([pcs, c_scaled], axis=1)

        km = KMeans(n_clusters=n_domains, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        return labels

    # ------------------------------------------------------------------ #
    # Convenience: auto-generate synthetic L-R pairs                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def make_synthetic_lr_pairs(cls, n_genes: int, n_pairs: int = 5) -> List[LRPair]:
        """Generate random L-R pairs for testing / demo purposes."""
        rng = np.random.default_rng(42)
        pairs = []
        used: set = set()
        while len(pairs) < n_pairs:
            l_idx = int(rng.integers(0, n_genes))
            r_idx = int(rng.integers(0, n_genes))
            if l_idx != r_idx and (l_idx, r_idx) not in used:
                used.add((l_idx, r_idx))
                pairs.append(LRPair(
                    ligand_idx=l_idx,
                    receptor_idx=r_idx,
                    name=f"L{l_idx}_R{r_idx}",
                ))
        return pairs
