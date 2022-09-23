# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import pint
import xarray as xr
import dask

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