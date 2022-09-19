# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from .array_index import ArrayIndex
from .bin_edge_array import BinEdgeArray
from .multi_mask_array import MultiMaskArray
from .uncertain_array import UncertainArray
import xarray as xr
import numpy as np
import pint

default_unit = object()

# This option ensures that we can use DataArray.sel with a Quantity created from a
# Python scalar, without having to manually make it a 0-D ndarray.
ureg = pint.UnitRegistry(force_ndarray_like=True)
Unit = ureg.Unit
Quantity = ureg.Quantity


def _units_for_dtype(units, dtype):
    if units is not default_unit:
        return units
    if dtype == bool:  # TODO proper check
        return None
    return ''


def linspace(dim, start, stop, num, *, endpoint=True, units=default_unit, dtype=None):
    data = np.linspace(start, stop, num, endpoint=endpoint, dtype=dtype)
    if (units := _units_for_dtype(units, dtype)) is not None:
        data = Quantity(data, units=units)
    return xr.DataArray(dims=(dim, ), data=data)


def as_edges(da, dim=None):
    dim = dim if dim is not None else da.dims[-1]
    edges = BinEdgeArray(da.values)
    if isinstance(da.data, Quantity):
        edges = Quantity(edges, units=da.data.units)
    return xr.DataArray(dims=da.dims, data=edges)


def array(dims, values, *, variances=None, units=default_unit, coords=None, masks=None):
    units = _units_for_dtype(units, values.dtype)
    data = values if variances is None else UncertainArray(values, variances)
    data = MultiMaskArray(data, masks=masks if masks is not None else {})
    data = data if units is None else Quantity(data, units)
    data = xr.Variable(dims=dims, data=data)
    coords = {} if coords is None else coords
    coords = {
        name: coord if isinstance(coord, xr.Variable) else coord.variable
        for name, coord in coords.items()
    }
    # Set coords without indexes to avoid stripping of units
    da = xr.DataArray(data=data, coords=coords, indexes={}, fastpath=True)
    for name in coords:
        da = da.set_xindex(name, ArrayIndex)
    return da


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

    def __getitem__(self, key):
        dim, val = key
        if isinstance(val, Quantity):
            coord = self._obj.coords[dim].data
            if val.units != coord.units:
                raise KeyError("Wrong unit")
            return self._obj[{dim: np.flatnonzero(coord.magnitude == val.magnitude)}]
