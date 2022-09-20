import numpy as np
import scippx as sx
import pytest


def test_basics():
    vectors = sx.VectorArray(np.arange(12).reshape(3, 4), ['x', 'y', 'z'])
    assert vectors.shape == (4, )


def test_add():
    vec1 = sx.VectorArray(np.array([1, 2, 3]), ['x', 'y', 'z'])
    vec2 = sx.VectorArray(np.array([3, 4, 5]), ['x', 'y', 'z'])
    vec = vec1 + vec2
    np.testing.assert_array_equal(vec.values, [4, 6, 8])


def test_add_fails_if_different_vectors():
    vec1 = sx.VectorArray(np.array([1, 2, 3]), ['x', 'y', 'z'])
    vec2 = sx.VectorArray(np.array([3, 4, 5]), ['vx', 'vy', 'vz'])
    with pytest.raises(ValueError):
        vec1 + vec2
