import numpy as np
import scippx as sx
import pytest
import xarray as xr
import pint
import dask
from numpy.testing import assert_array_equal
from scippx import array_property
from scippx.array_property import Unit, Quantity


def test_basics():
    vectors = sx.VectorArray(np.arange(15).reshape(5, 3), ['vx', 'vy', 'vz'])
    edges = sx.BinEdgeArray(vectors)
    data = Quantity(edges, 'meter/second')
    da = xr.DataArray(dims=('x', ), data=data, coords={'x': np.arange(4)})
    assert da.units == Unit('m/s')
    assert da.left.units == Unit('m/s')
    assert da.fields['vx'].units == Unit('m/s')
    assert da.fields['vx'].left.units == Unit('m/s')
    assert da.left.fields['vx'].units == Unit('m/s')
    da.left
    da.left.fields['vx']
    da.fields['vx'].left
    da.fields['vx'].left.units
    da.fields['vx'].left.coords['x']
    da.fields['vx'].magnitude.left
    with pytest.raises(AttributeError):
        da.left.left


def test_method():
    vectors = sx.VectorArray(np.arange(15).reshape(5, 3), ['vx', 'vy', 'vz'])
    edges = sx.BinEdgeArray(vectors)
    data = Quantity(edges, 'meter/second')
    da = xr.DataArray(dims=('x', ), data=data, coords={'x': np.arange(4)})
    assert da.center().units == Unit('m/s')
    da.center().fields['vy'].units
    da.magnitude.center()


def test_mask_array():
    vectors = sx.VectorArray(np.arange(15).reshape(5, 3), ['vx', 'vy', 'vz'])
    edges = sx.BinEdgeArray(vectors)
    data = Quantity(edges, 'meter/second')
    masked = sx.MultiMaskArray(data,
                               masks={'mask1': np.array([False, False, True, False])})
    da = xr.DataArray(dims=('x', ), data=masked, coords={'x': np.arange(4)})

    # Note how neither `mask1` nor MultiMaskArray know about dims, but this works:
    assert da.masks['mask1'].dims == ('x', )
    # Note the order: BinEdgeArray is *inside* MultiMaskArray, but we get the mask
    da.left.masks['mask1']
    assert da.units == Unit('m/s')
    assert not hasattr(da.masks['mask1'], 'units')


def test_mask_array_masks_setitem():
    vectors = sx.VectorArray(np.arange(15).reshape(5, 3), ['vx', 'vy', 'vz'])
    edges = sx.BinEdgeArray(vectors)
    data = Quantity(edges, 'meter/second')
    masked = sx.MultiMaskArray(data,
                               masks={'mask1': np.array([False, False, True, False])})
    da = xr.DataArray(dims=('x', ), data=masked, coords={'x': np.arange(4)})

    with pytest.raises(ValueError):  # Not a xr.Variable
        da.masks['new_mask'] = np.array([False, False, True, False])

    with pytest.raises(ValueError):  # Bad dims
        da.masks['new_mask'] = xr.Variable(dims=('x2', ),
                                           data=np.array([False, False, True, False]))
    assert len(da.masks) == 1
    da.masks['new_mask'] = xr.Variable(dims=('x', ),
                                       data=np.array([False, False, True, False]))
    assert len(da.masks) == 2


def test_setattr():
    vectors = sx.VectorArray(np.arange(15).reshape(5, 3), ['vx', 'vy', 'vz'])
    edges = sx.BinEdgeArray(vectors)
    data = Quantity(edges, 'meter/second')
    masked = sx.MultiMaskArray(data,
                               masks={'mask1': np.array([False, False, True, False])})
    da = xr.DataArray(dims=('x', ), data=masked, coords={'x': np.arange(4)})

    # TODO We probably need a separate __array_setattr__ for this
    # da.magnitude += 2


def test_dask():
    array = dask.array.arange(15).reshape(5, 3)
    vectors = sx.VectorArray(array, ['vx', 'vy', 'vz'])
    edges = sx.BinEdgeArray(vectors)
    data = Quantity(edges, 'meter/second')
    masked = sx.MultiMaskArray(data,
                               masks={'mask1': np.array([False, False, True, False])})
    da = xr.DataArray(dims=('x', ), data=masked, coords={'x': np.arange(4)})
    result = da.xcompute()
    assert result.dims == ('x', )
    assert result.units == Unit('m/s')
    assert 'mask1' in da.masks


# np.empty(...., like=...) cannot cope with nesting (tried edit in
# dask/array/core.py:5288). Changing to use np.empy_like works!
def test_dask_chunked_masks():
    array = np.arange(15).reshape(5, 3)
    vectors = sx.VectorArray(array, ['vx', 'vy', 'vz'])
    edges = sx.BinEdgeArray(vectors)
    data = Quantity(edges, 'meter/second')
    masked = sx.MultiMaskArray(data,
                               masks={'mask1': np.array([False, False, True, False])})
    chunked = dask.array.from_array(masked, chunks=(2, ), asarray=False)
    da = xr.DataArray(dims=('x', ), data=chunked, coords={'x': np.arange(4)})
    da = da + da
    result = da.xcompute()
    assert result.dims == ('x', )
    # TODO
    # pint does not implement NEP-18 so np.empty_like strips the unit
    # assert result.units == Unit('m/s')
    assert 'mask1' in result.masks
    assert_array_equal(result.left.fields['vx'].values, [0, 2, 4, 6])
    assert_array_equal(result.right.fields['vz'].values, [22, 24, 26, 28])
