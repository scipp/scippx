import numpy as np
import scippx as sx
import pytest


@pytest.fixture
def list_array_1d():
    starts = np.array([0, 2])
    stops = np.array([2, 3])
    content = np.array([0.1, 0.2, 0.3])
    return sx.ListArray(starts=starts, stops=stops, content=content)


def test_create():
    starts = np.array([0, 2])
    stops = np.array([2, 3])
    content = np.array([0.1, 0.2, 0.3])
    la = sx.ListArray(starts=starts, stops=stops, content=content)
    assert la.shape == (2, )


def test_values(list_array_1d):
    np.testing.assert_array_equal(list_array_1d[0].values, [0.1, 0.2])
    np.testing.assert_array_equal(list_array_1d[1].values, [0.3])
