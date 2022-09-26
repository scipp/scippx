import numpy as np
import scippx as sx
import pytest
from scippx.array_property import Quantity, Unit
from numpy.testing import assert_array_equal


def xtest_callable_content_array_attr():
    data = sx.BinEdgeArray(np.arange(3))
    mask = np.array([True, False])
    edge_mask = sx.BinEdgeArray(np.array([False, False, True]))
    mma = sx.MultiMaskArray(data, masks={'mask': mask, 'edge_mask': edge_mask})
    result = mma.center()
    assert_array_equal(result.data, [0.5, 1.5])
    assert_array_equal(result.masks['mask'], [True, False])
    assert_array_equal(result.masks['edge_mask'], [False, True])


def test_heterogenous_content_array_attr():
    data = sx.BinEdgeArray(np.arange(3))
    mask = np.array([True, False])
    edge_mask = sx.BinEdgeArray(np.array([False, False, True]))
    mma = sx.MultiMaskArray(data, masks={'mask': mask, 'edge_mask': edge_mask})
    left = mma.left
    assert_array_equal(left.data, [0, 1])
    assert_array_equal(left.masks['mask'], [True, False])
    assert_array_equal(left.masks['edge_mask'], [False, False])


def test_heterogenous_content_array_attr_only_on_data():
    data = Quantity(np.arange(2), 'm')
    mask = np.array([True, False])
    mma = sx.MultiMaskArray(data, masks={'mask': mask})
    units = mma.units
    assert isinstance(units, Unit)
