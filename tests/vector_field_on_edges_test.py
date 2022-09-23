import numpy as np
import scippx as sx
import pytest
import xarray as xr


def test_basics():
    vectors = sx.VectorArray(np.arange(15).reshape(5, 3), ['vx', 'vy', 'vz'])
    edges = sx.BinEdgeArray(vectors)
    da = xr.DataArray(dims=('x', ), data=edges, coords={'x': np.arange(4)})
    da += 2
    da + da
    np.testing.assert_array_equal(da.data.left.fields['vy'], [3, 6, 9, 12])
    np.testing.assert_array_equal(da.data.right.fields['vy'], [6, 9, 12, 15])
    np.testing.assert_array_equal(da.data.values.fields['vy'], [3, 6, 9, 12, 15])
