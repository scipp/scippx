import numpy as np
import scippx as sx
import pytest
import xarray as xr
import pint
import dask
from numpy.testing import assert_array_equal

ureg = pint.UnitRegistry(force_ndarray_like=True)
Unit = ureg.Unit
Quantity = ureg.Quantity


def _quantity_array_property(self, name, wrap, unwrap):
    if name == 'units':
        return self.units
    if name == 'magnitude':
        return wrap(self.magnitude)
    if hasattr(self.magnitude, '__array_property__'):
        wrap_ = lambda x: wrap(self.__class__(x, self.units))
        unwrap_ = unwrap  # TODO strip and handle units
        return self.magnitude.__array_property__(name, wrap=wrap_, unwrap=unwrap_)
    raise AttributeError(f"{self.__class__} object has no attribute '{name}'")


setattr(Quantity, '__array_property__', _quantity_array_property)

# props accessor is a partial solution, but makes chaining cumbersome:
#   da.props.left.props.fields['vx']
#
# @xr.register_dataarray_accessor('props')
# class PropertyAccessor:
#
#     def __init__(self, xarray_obj):
#         self._obj = xarray_obj
#
#     def __getattr__(self, name: str):
#
#         def wrap(x):
#             out = self._obj.copy(deep=False)
#             out.data = x
#             return out
#
#         return self._obj.data.__array_property__(name, wrap=wrap)


def _dataarray_array_property(self, name, wrap, unwrap):
    # TODO Could use this mechanism to handle coords
    def wrap_(x):
        out = self.copy(deep=False)
        out.data = x
        return wrap(out)

    def unwrap_(x):
        x = unwrap(x)
        if isinstance(x, (xr.DataArray, xr.Variable)):
            # TODO check coords
            if x.dims != self.dims:  # TODO support broadcast/transpose/...
                raise ValueError("Incompatible dims {x.dims} with {self.dims}")
            return x.data
        # TODO Could accept anything with dims property!
        raise ValueError("Expected xr.DataArray or xr.Variable")

    if hasattr(self.data, '__array_property__'):
        return self.data.__array_property__(name, wrap=wrap_, unwrap=unwrap_)
    raise AttributeError(f"{self.__class__} object has no attribute '{name}'")


def _dataarray_getattr(self, name: str):
    # Top-level, wrap is no-op
    return self.__array_property__(name, wrap=lambda x: x, unwrap=lambda x: x)


# This is not a nice thing to do, unless this becomes a generally agreed upon solution,
# i.e., xarray would ship with this.
setattr(xr.DataArray, '__array_property__', _dataarray_array_property)
setattr(xr.DataArray, '__getattr__', _dataarray_getattr)


def _dask_array_property(self, name, wrap, unwrap):
    if name == 'xcompute':
        from scippx.array_attr import rewrap_result
        return rewrap_result(wrap)(self.compute)
    raise AttributeError(f"{self.__class__} object has no attribute '{name}'")


setattr(dask.array.core.Array, '__array_property__', _dask_array_property)


def test_basics():
    vectors = sx.VectorArray(np.arange(15).reshape(3, 5), ['vx', 'vy', 'vz'])
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
    vectors = sx.VectorArray(np.arange(15).reshape(3, 5), ['vx', 'vy', 'vz'])
    edges = sx.BinEdgeArray(vectors)
    data = Quantity(edges, 'meter/second')
    da = xr.DataArray(dims=('x', ), data=data, coords={'x': np.arange(4)})
    assert da.center().units == Unit('m/s')
    da.center().fields['vy'].units
    da.magnitude.center()


def test_mask_array():
    vectors = sx.VectorArray(np.arange(15).reshape(3, 5), ['vx', 'vy', 'vz'])
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
    vectors = sx.VectorArray(np.arange(15).reshape(3, 5), ['vx', 'vy', 'vz'])
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
    vectors = sx.VectorArray(np.arange(15).reshape(3, 5), ['vx', 'vy', 'vz'])
    edges = sx.BinEdgeArray(vectors)
    data = Quantity(edges, 'meter/second')
    masked = sx.MultiMaskArray(data,
                               masks={'mask1': np.array([False, False, True, False])})
    da = xr.DataArray(dims=('x', ), data=masked, coords={'x': np.arange(4)})

    # TODO We probably need a separate __array_setattr__ for this
    # da.magnitude += 2


def test_dask():
    array = dask.array.arange(15).reshape(3, 5)
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
    array = np.arange(15).reshape(3, 5)
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
