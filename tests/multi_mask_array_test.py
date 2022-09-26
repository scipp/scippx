import numpy as np
import scippx as sx
import pytest
from numpy.testing import assert_array_equal


def test_callable_content_array_attr():
    data = sx.BinEdgeArray(np.arange(3))
    mask = np.array([True, False])
    edge_mask = sx.BinEdgeArray(np.array([False, False, True]))
    mma = sx.MultiMaskArray(data, masks={'mask': mask, 'edge_mask': edge_mask})
    result = mma.center()
    assert_array_equal(result.data, [0.5, 1.5])
    assert_array_equal(result.masks['mask'], [True, False])
    assert_array_equal(result.masks['edge_mask'], [False, True])
