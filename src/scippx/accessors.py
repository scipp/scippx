# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Dict
import xarray as xr
import pint


@xr.register_dataarray_accessor('masks')
class MaskAccessor:

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __getitem__(self, name: str):
        return xr.Variable(dims=self._obj.dims, data=self._obj.data.masks[name])


@xr.register_dataarray_accessor('qsel')
class IndexAccessor:

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __getitem__(self, sel: Dict[str, pint.Quantity]):
        for dim, q in sel.items():
            if self._obj.coords[dim].data.units != q.units:
                raise KeyError("Key has wrong unit.")
        return self._obj.loc[{dim: q.magnitude for dim, q in sel.items()}]
