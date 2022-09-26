import numpy as np
import scippx as sx
import pytest
from numpy.testing import assert_array_equal


def test_basics():
    edges = sx.BinEdgeArray(np.arange(5))
    assert len(edges) == 4
    assert edges.ndim == 1
    assert edges.shape == (4, )
    assert_array_equal(edges.left, np.arange(4))
    assert_array_equal(edges.right, np.arange(1, 5))


def test_getitem():
    edges = sx.BinEdgeArray(np.arange(5))
    s = edges[1:3]
    assert len(s) == 2
    assert_array_equal(s.edges, [1, 2, 3])


def test_ufuncs():
    edges = sx.BinEdgeArray(np.arange(5))
    edges += 1
    assert_array_equal(edges.edges, np.arange(1, 6))


def test_ufuncs_raise_with_ndarray():
    edges = sx.BinEdgeArray(np.arange(5))
    with pytest.raises(AttributeError):
        edges + np.arange(5)
