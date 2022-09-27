import numpy as np
import scippx as sx
import pytest
from scippx.array_property import Unit, Quantity


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


def test_add_raises_with_ndarray_holding_vector():
    vec1 = sx.VectorArray(np.array([[1, 2, 3], [4, 5, 6]]), ['x', 'y', 'z'])
    vec2 = np.array([[1, 1, 1], [2, 2, 2]])
    with pytest.raises(ValueError):
        vec1 + vec2


def test_add_raises_with_ndarray_holding_scalar():
    vec1 = sx.VectorArray(np.array([[1, 2, 3], [4, 5, 6]]), ['x', 'y', 'z'])
    scalar = np.array([1, 2])
    with pytest.raises(ValueError):
        vec1 + scalar


def test_mul_works_with_ndarray_holding_scalar():
    vec = sx.VectorArray(np.array([[1, 2, 3], [4, 5, 6]]), ['x', 'y', 'z'])
    scalar = np.array([1, 2])
    result = vec * scalar
    np.testing.assert_array_equal(result.values, [[1, 2, 3], [8, 10, 12]])


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


def test_dot_quantity_of_vector():
    elems = np.array([[1, 2, 3], [4, 5, 6]])
    vec = sx.VectorArray(elems, ['x', 'y', 'z'])
    vec =  Quantity(vec, 'm')
    result = np.dot(vec, vec)
    np.testing.assert_array_equal(result, [1 + 4 + 9, 16 + 25 + 36])
    assert result.units == Unit('m**2')


def test_dot_vector_of_quantity():
    elems = np.array([[1, 2, 3], [4, 5, 6]])
    elems =  Quantity(elems, 'm')
    vec = sx.VectorArray(elems, ['x', 'y', 'z'])
    result = np.dot(vec, vec)
    np.testing.assert_array_equal(result, [1 + 4 + 9, 16 + 25 + 36])
    assert result.units == Unit('m**2')


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


def test_getitem_ellipsis():
    vectors = sx.VectorArray(np.array([[1, 2, 3], [4, 5, 6]]), ['x', 'y', 'z'])
    np.testing.assert_array_equal(vectors[..., 1], [4, 5, 6])
