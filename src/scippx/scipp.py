# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from .multi_mask_array import MultiMaskArray
from .uncertain_array import UncertainArray
import xarray as xr
import numpy as np
import pint

default_unit = object()


def _units_for_dtype(units, dtype):
    if units is not default_unit:
        return units
    if dtype == bool:  # TODO proper check
        return None
    return ''


def linspace(dim, start, stop, num, *, endpoint=True, units=default_unit, dtype=None):
    data = np.linspace(start, stop, num, endpoint=endpoint, dtype=dtype)
    if (units := _units_for_dtype(units, dtype)) is not None:
        data = pint.Quantity(data, units=units)
    return xr.DataArray(dims=(dim, ), data=data)


def array(dims, values, *, variances=None, units=default_unit, coords=None, masks=None):
    units = _units_for_dtype(units, values.dtype)
    data = values if variances is None else UncertainArray(values, variances)
    data = MultiMaskArray(data, masks=masks if masks is not None else {})
    data = data if units is None else pint.Quantity(data, units)
    # Build indexes
    tmp = xr.DataArray(dims=dims, data=data, coords=coords)
    # Set coords with custom indexes to avoid stripping of units
    return xr.DataArray(data=tmp.variable,
                        coords={} if coords is None else coords,
                        indexes=tmp.indexes,
                        fastpath=True)



class Coords:

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __getitem__(self, name: str):
        # Wrap in DataArray to avoid returning index, which has weird behavior
        return xr.DataArray(self._obj.coords[name].variable)

    def __setitem__(self, name: str, value):
        self._obj.coords[name] = value


@xr.register_dataarray_accessor('scipp')
class Scipp:

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def coords(self):
        return Coords(self._obj)
