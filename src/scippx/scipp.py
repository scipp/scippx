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
    return xr.DataArray(dims=dims, data=data, coords=coords)
