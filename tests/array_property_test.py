import numpy as np
import scippx as sx
import pytest
import xarray as xr
import pint

ureg = pint.UnitRegistry(force_ndarray_like=True)
Unit = ureg.Unit
Quantity = ureg.Quantity


def _quantity_array_property(self, name, wrap):
    if name == 'units':
        return self.units
    if name == 'magnitude':
        return wrap(self.magnitude)
    if hasattr(self.magnitude, '__array_property__'):
        return self.magnitude.__array_property__(
            name, wrap=lambda x: wrap(self.__class__(x, self.units)))
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


def _dataarray_array_property(self, name, wrap):
    # TODO Could use this mechanism to handle coords
    def wrap(x):
        out = self.copy(deep=False)
        out.data = x
        return out

    if hasattr(self.data, '__array_property__'):
        return self.data.__array_property__(name, wrap=wrap)
    raise AttributeError(f"{self.__class__} object has no attribute '{name}'")


def _dataarray_getattr(self, name: str):
    # Top-level, wrap is no-op
    return self.__array_property__(name, wrap=lambda x: x)


# This is not a nice thing to do, unless this becomes a generally agreed upon solution,
# i.e., xarray would ship with this.
setattr(xr.DataArray, '__array_property__', _dataarray_array_property)
setattr(xr.DataArray, '__getattr__', _dataarray_getattr)


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
