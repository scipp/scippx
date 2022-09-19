from copy import copy
import numpy as np
import scippx as sx
import pytest


def make_list_array_1d(values=None):
    starts = np.array([0, 2])
    stops = np.array([2, 3])
    content = np.array([0.1, 0.2, 0.3] if values is None else values)
    return sx.ListArray(starts=starts, stops=stops, content=content)


def make_list_array_2d(values=None):
    starts = np.array([[0, 2], [3, 3]])
    stops = np.array([[2, 3], [3, 4]])
    content = np.array([0.1, 0.2, 0.3, 0.4] if values is None else values)
    return sx.ListArray(starts=starts, stops=stops, content=content)


@pytest.fixture
def list_array_1d():
    return make_list_array_1d()


@pytest.fixture
def list_array_2d():
    return make_list_array_2d()


def test_create():
    starts = np.array([0, 2])
    stops = np.array([2, 3])
    content = np.array([0.1, 0.2, 0.3])
    la = sx.ListArray(starts=starts, stops=stops, content=content)
    assert la.shape == (2, )


def test_values(list_array_1d):
    np.testing.assert_array_equal(list_array_1d[0].values, [0.1, 0.2])
    np.testing.assert_array_equal(list_array_1d[1].values, [0.3])


def test_setitem_ellipsis():
    la1 = make_list_array_1d([1, 2, 3])
    la2 = make_list_array_1d([3, 4, 5])
    la1[...] = la2
    np.testing.assert_array_equal(la1._content, [3, 4, 5])
    la1[1] = la2[1]


def test_setitem_single():
    la1 = make_list_array_1d([1, 2, 3])
    la2 = make_list_array_1d([3, 4, 5])
    la1[1] = la2[1]
    np.testing.assert_array_equal(la1._content, [1, 2, 5])


def test_setitem_range():
    la1 = make_list_array_1d([1, 2, 3])
    la2 = make_list_array_1d([3, 4, 5])
    la1[0:1] = la2[0:1]
    np.testing.assert_array_equal(la1._content, [3, 4, 3])


def test_setitem_size_mistmatch():
    la1 = make_list_array_1d([1, 2, 3])
    la2 = make_list_array_1d([3, 4, 5])
    with pytest.raises(ValueError):
        la1[0] = la2[1]
    np.testing.assert_array_equal(la1._content, [1, 2, 3])


def test_copy(list_array_2d):
    la = list_array_2d
    result = copy(la)
    np.testing.assert_array_equal(result._starts, la._starts)
    np.testing.assert_array_equal(result._stops, la._stops)
    np.testing.assert_array_equal(result._content, la._content)
    result = copy(la[1])
    np.testing.assert_array_equal(result._starts, [0, 0])
    np.testing.assert_array_equal(result._stops, [0, 1])
    np.testing.assert_array_equal(result._content, [0.4])
