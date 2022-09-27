# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import pint
import xarray as xr
import dask
from .array_index import ArrayIndex
from .array_attr import ArrayAccessor

ureg = pint.UnitRegistry(force_ndarray_like=True)
Unit = ureg.Unit
Quantity = ureg.Quantity

class QuantityAccessor:
    def __init__(self, quantity, wrap, unwrap):
        self._quantity = quantity
        self._wrap = wrap
        self._unwrap = unwrap

    def __getattr__(self, attr):
        return rewrap_result(self._wrap)(getattr(self._quantity, attr))

def _quantity_array_getattr(self, name, wrap, unwrap):
    if name == 'units':
        return self.units
    if name == 'magnitude':
        return wrap(self.magnitude)
    if name == 'quantity':
        return ArrayAccessor(self, wrap, unwrap)
    if hasattr(self.magnitude, '__array_getattr__'):
        wrap_ = lambda x: wrap(self.__class__(x, self.units))
        unwrap_ = unwrap  # TODO strip and handle units
        return self.magnitude.__array_getattr__(name, wrap=wrap_, unwrap=unwrap_)
    raise AttributeError(f"{self.__class__} object has no attribute '{name}'")


setattr(Quantity, '__array_getattr__', _quantity_array_getattr)

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
#         return self._obj.data.__array_getattr__(name, wrap=wrap)


def _dataarray_array_getattr(self, name, wrap, unwrap):
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

    if hasattr(self.data, '__array_getattr__'):
        return self.data.__array_getattr__(name, wrap=wrap_, unwrap=unwrap_)
    raise AttributeError(f"{self.__class__} object has no attribute '{name}'")


def _dataarray_getattr(self, name: str):
    # Top-level, wrap is no-op
    return self.__array_getattr__(name, wrap=lambda x: x, unwrap=lambda x: x)


# This is not a nice thing to do, unless this becomes a generally agreed upon solution,
# i.e., xarray would ship with this.
setattr(xr.DataArray, '__array_getattr__', _dataarray_array_getattr)
setattr(xr.DataArray, '__getattr__', _dataarray_getattr)


def _dask_array_property(self, name, wrap, unwrap):
    if name == 'xcompute':
        from scippx.array_attr import rewrap_result
        return rewrap_result(wrap)(self.compute)
    raise AttributeError(f"{self.__class__} object has no attribute '{name}'")


setattr(dask.array.core.Array, '__array_getattr__', _dask_array_property)


def DataArray(*, dims, data, coords):
    coords = {
        key: xr.Variable(dims=(key, ), data=values)
        for key, values in coords.items()
    }
    var = xr.Variable(dims=dims, data=data)
    da = xr.DataArray(var, coords=coords, indexes={}, fastpath=True)
    for dim in list(da.dims):
        da = da.set_xindex(dim, ArrayIndex)
    return da
