"""Tests for SpatialGraphTransformer."""
import numpy as np
import torch
import pytest
from spatial_omics_gfm import SpatialGraph, SpatialGraphTransformer, SpatialSpot


def make_graph(n=20, n_genes=64, k=4, seed=1):
    rng = np.random.default_rng(seed)
    spots = []
    for i in range(n):
        expr = rng.exponential(size=n_genes).astype(np.float32)
        spots.append(SpatialSpot(x=float(i % 5), y=float(i // 5), expression=expr, spot_id=f"s{i}"))
    return SpatialGraph(spots, k=k)


def test_from_graph_builds():
    g = make_graph()
    m = SpatialGraphTransformer.from_graph(g, hidden_dim=64, num_layers=2, num_heads=4)
    assert m.n_genes == g.n_genes
    assert m.hidden_dim == 64


def test_forward_shapes():
    g = make_graph(n=15)
    m = SpatialGraphTransformer.from_graph(g, hidden_dim=32, num_layers=2, num_heads=4, n_classes=3)
    emb, logits = m(g.node_features, g.coords, g.edge_index, g.edge_attr)
    assert emb.shape == (15, 32)
    assert logits is not None and logits.shape == (15, 3)


def test_forward_no_classifier():
    g = make_graph(n=10)
    m = SpatialGraphTransformer.from_graph(g, hidden_dim=32, num_layers=1, num_heads=4)
    emb, logits = m(g.node_features, g.coords, g.edge_index, g.edge_attr)
    assert emb.shape == (10, 32)
    assert logits is None


def test_encode_deterministic():
    g = make_graph(n=10)
    m = SpatialGraphTransformer.from_graph(g, hidden_dim=32, num_layers=2, num_heads=4)
    m.eval()
    with torch.no_grad():
        e1 = m.encode(g.node_features, g.coords, g.edge_index, g.edge_attr)
        e2 = m.encode(g.node_features, g.coords, g.edge_index, g.edge_attr)
    assert torch.allclose(e1, e2)


def test_parameter_count_positive():
    g = make_graph()
    m = SpatialGraphTransformer.from_graph(g, hidden_dim=64, num_layers=2)
    assert m.parameter_count() > 0


def test_gradients_flow():
    g = make_graph(n=8)
    m = SpatialGraphTransformer.from_graph(g, hidden_dim=32, num_layers=2, num_heads=4, n_classes=4)
    emb, logits = m(g.node_features, g.coords, g.edge_index, g.edge_attr)
    loss = logits.sum()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
