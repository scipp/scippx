# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import xarray as xr


@xr.register_dataarray_accessor('masks')
class MaskAccessor:

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __getitem__(self, name: str):
        return xr.Variable(dims=self._obj.dims, data=self._obj.data.masks[name])
