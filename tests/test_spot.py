"""Tests for SpatialSpot."""
import numpy as np
import pytest
from spatial_omics_gfm import SpatialSpot


def make_spot(n_genes=100, x=1.0, y=2.0, **kw):
    expr = np.random.default_rng(0).exponential(size=n_genes).astype(np.float32)
    return SpatialSpot(x=x, y=y, expression=expr, **kw)


def test_basic_attrs():
    s = make_spot(spot_id="bc01", cell_type="Tumor")
    assert s.x == 1.0
    assert s.y == 2.0
    assert s.n_genes == 100
    assert s.spot_id == "bc01"
    assert s.cell_type == "Tumor"


def test_coords_shape():
    s = make_spot()
    c = s.coords
    assert c.shape == (2,)
    assert c[0] == s.x and c[1] == s.y


def test_total_counts_nonneg():
    s = make_spot()
    assert s.total_counts >= 0


def test_log_normalize_shape():
    s = make_spot(n_genes=50)
    ln = s.log_normalize()
    assert ln.shape == (50,)
    assert ln.dtype == np.float32


def test_log_normalize_zero_counts():
    s = SpatialSpot(x=0, y=0, expression=np.zeros(10, dtype=np.float32))
    ln = s.log_normalize()
    assert (ln == 0).all()


def test_normalized_expression_nonneg():
    s = make_spot()
    ne = s.normalized_expression()
    assert (ne >= 0).all()


def test_repr():
    s = make_spot(spot_id="foo")
    assert "foo" in repr(s)
