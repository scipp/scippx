import numpy as np
import scippx as sx
import pytest


def test_basics():
    vectors = sx.VectorArray(np.arange(12).reshape(4, 3), ['x', 'y', 'z'])
    assert vectors.shape == (4, )


def test_add():
    vec1 = sx.VectorArray(np.array([1, 2, 3]), ['x', 'y', 'z'])
    vec2 = sx.VectorArray(np.array([3, 4, 5]), ['x', 'y', 'z'])
    vec = vec1 + vec2
    np.testing.assert_array_equal(vec.values, [4, 6, 8])


def test_add_raises_if_different_vectors():
    vec1 = sx.VectorArray(np.array([1, 2, 3]), ['x', 'y', 'z'])
    vec2 = sx.VectorArray(np.array([3, 4, 5]), ['vx', 'vy', 'vz'])
    with pytest.raises(ValueError):
        vec1 + vec2


def test_mul_raises():
    vec = sx.VectorArray(np.array([1, 2, 3]), ['x', 'y', 'z'])
    with pytest.raises(ValueError):
        vec * vec


def test_scale():
    vec = sx.VectorArray(np.array([1, 2, 3]), ['x', 'y', 'z'])
    result = 3 * vec
    np.testing.assert_array_equal(result.values, [3, 6, 9])


def test_dot():
    vec = sx.VectorArray(np.array([[1, 2, 3], [4, 5, 6]]), ['x', 'y', 'z'])
    result = np.dot(vec, vec)
    np.testing.assert_array_equal(result, [1 + 4 + 9, 16 + 25 + 36])


def test_setitem():
    vec1 = sx.VectorArray(np.array([[1, 2, 3], [1, 2, 3]]), ['x', 'y', 'z'])
    vec2 = sx.VectorArray(np.array([4, 5, 6]), ['x', 'y', 'z'])
    vec1[1] = vec2
    np.testing.assert_array_equal(vec1.values, [[1, 2, 3], [4, 5, 6]])


def test_setitem_incompatible_field_names_raises_ValueError():
    vec1 = sx.VectorArray(np.array([[1, 2, 3], [4, 5, 6]]), ['x', 'y', 'z'])
    vec2 = sx.VectorArray(np.array([[1, 2, 3], [4, 5, 6]]), ['u', 'v', 'w'])
    with pytest.raises(ValueError):
        vec1[...] = vec2
