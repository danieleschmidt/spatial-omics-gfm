"""Tests for CellCommunicationInferer."""
import numpy as np
import torch
import pytest
from spatial_omics_gfm import (
    SpatialGraph, SpatialSpot,
    CellCommunicationInferer, LRPair,
)


def make_graph(n=20, n_genes=50, k=4, seed=2):
    rng = np.random.default_rng(seed)
    spots = []
    for i in range(n):
        expr = rng.exponential(size=n_genes).astype(np.float32)
        spots.append(SpatialSpot(x=float(i % 5), y=float(i // 5), expression=expr, spot_id=f"s{i}"))
    return SpatialGraph(spots, k=k)


def test_infer_returns_results():
    g = make_graph()
    lr_pairs = CellCommunicationInferer.make_synthetic_lr_pairs(50, n_pairs=3)
    infer = CellCommunicationInferer(lr_pairs)
    results = infer.infer(g.node_features, g.edge_index, g.edge_attr)
    assert len(results) == 3


def test_result_shapes():
    g = make_graph(n=15)
    lr_pairs = [LRPair(0, 1, "L0_R1")]
    infer = CellCommunicationInferer(lr_pairs)
    results = infer.infer(g.node_features, g.edge_index, g.edge_attr)
    r = results[0]
    assert len(r.scores) == g.num_edges
    assert len(r.sender_idx) == g.num_edges


def test_scores_nonneg():
    g = make_graph()
    lr = [LRPair(0, 5, "test")]
    infer = CellCommunicationInferer(lr)
    res = infer.infer(g.node_features, g.edge_index, g.edge_attr)[0]
    assert (res.scores >= 0).all()


def test_top_interactions_length():
    g = make_graph(n=12)
    lr = [LRPair(0, 1)]
    infer = CellCommunicationInferer(lr)
    res = infer.infer(g.node_features, g.edge_index, g.edge_attr)[0]
    top = res.top_interactions(n=5)
    assert len(top) == 5


def test_pairwise_correlation_shape():
    g = make_graph(n=15)
    corr = CellCommunicationInferer.pairwise_correlation(
        g.node_features, g.edge_index, top_genes=20
    )
    assert corr.shape == (g.num_edges,)


def test_pairwise_correlation_range():
    g = make_graph(n=20)
    corr = CellCommunicationInferer.pairwise_correlation(g.node_features, g.edge_index)
    assert corr.min() >= -1.01 and corr.max() <= 1.01


def test_detect_spatial_domains_labels():
    g = make_graph(n=30)
    labels = CellCommunicationInferer.detect_spatial_domains(
        g.node_features, g.coords, n_domains=4
    )
    assert labels.shape == (30,)
    assert set(labels).issubset(set(range(4)))


def test_make_synthetic_lr_pairs():
    pairs = CellCommunicationInferer.make_synthetic_lr_pairs(n_genes=100, n_pairs=5)
    assert len(pairs) == 5
    for p in pairs:
        assert p.ligand_idx != p.receptor_idx
