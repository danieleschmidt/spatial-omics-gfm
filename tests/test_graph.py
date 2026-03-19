"""Tests for SpatialGraph."""
import numpy as np
import torch
import pytest
from spatial_omics_gfm import SpatialSpot, SpatialGraph


def make_spots(n=10, n_genes=50, seed=0):
    rng = np.random.default_rng(seed)
    spots = []
    for i in range(n):
        expr = rng.exponential(size=n_genes).astype(np.float32)
        spots.append(SpatialSpot(
            x=float(i % 5),
            y=float(i // 5),
            expression=expr,
            spot_id=f"s{i}",
        ))
    return spots


def test_build_basic():
    spots = make_spots(20)
    g = SpatialGraph(spots, k=4)
    assert g.num_nodes == 20
    assert g.num_edges == 20 * 4   # directed, each node → 4 neighbours
    assert g.n_genes == 50


def test_node_features_shape():
    spots = make_spots(15)
    g = SpatialGraph(spots, k=3)
    assert g.node_features.shape == (15, 50)
    assert g.coords.shape == (15, 2)


def test_edge_index_bounds():
    spots = make_spots(12)
    g = SpatialGraph(spots, k=3)
    ei = g.edge_index
    assert ei.shape[0] == 2
    assert int(ei.min()) >= 0
    assert int(ei.max()) < 12


def test_edge_attr_shape():
    spots = make_spots(10)
    g = SpatialGraph(spots, k=4)
    assert g.edge_attr.shape == (g.num_edges, 2)


def test_adjacency_list_lengths():
    spots = make_spots(10)
    k = 3
    g = SpatialGraph(spots, k=k)
    adj = g.adjacency_list()
    assert len(adj) == 10
    for nbrs in adj:
        assert len(nbrs) == k


def test_empty_spots_raises():
    with pytest.raises(ValueError):
        SpatialGraph([], k=3)


def test_repr_contains_nodes():
    spots = make_spots(8)
    g = SpatialGraph(spots, k=2)
    assert "nodes=8" in repr(g)
