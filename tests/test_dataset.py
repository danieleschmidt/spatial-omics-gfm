"""Tests for SyntheticSpatialDataset."""
import numpy as np
import pytest
from spatial_omics_gfm import SyntheticSpatialDataset, SpatialSpot


def test_basic_construction():
    ds = SyntheticSpatialDataset(n_spots=25, n_genes=100, n_regions=3)
    assert len(ds) == 25


def test_spot_types():
    ds = SyntheticSpatialDataset(n_spots=20)
    assert all(isinstance(s, SpatialSpot) for s in ds.spots)


def test_expression_shape():
    ds = SyntheticSpatialDataset(n_spots=10, n_genes=200)
    for s in ds.spots:
        assert s.expression.shape == (200,)


def test_cell_types_populated():
    ds = SyntheticSpatialDataset(n_spots=30, n_regions=4)
    types = ds.get_cell_types()
    assert all(t is not None for t in types)
    assert len(ds.unique_cell_types()) >= 1


def test_coords_shape():
    ds = SyntheticSpatialDataset(n_spots=16)
    c = ds.get_coords()
    assert c.shape == (16, 2)


def test_unique_ids():
    ds = SyntheticSpatialDataset(n_spots=50)
    ids = [s.spot_id for s in ds.spots]
    assert len(set(ids)) == len(ids)


def test_nonneg_expression():
    ds = SyntheticSpatialDataset(n_spots=20, n_genes=50)
    for s in ds.spots:
        assert (s.expression >= 0).all()


def test_reproducible_with_seed():
    ds1 = SyntheticSpatialDataset(n_spots=30, seed=7)
    ds2 = SyntheticSpatialDataset(n_spots=30, seed=7)
    assert np.allclose(ds1[0].expression, ds2[0].expression)


def test_different_seeds_differ():
    ds1 = SyntheticSpatialDataset(n_spots=20, seed=0)
    ds2 = SyntheticSpatialDataset(n_spots=20, seed=99)
    assert not np.allclose(ds1[0].expression, ds2[0].expression)
