# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import numpy as np
import numpy.lib.mixins
from copy import copy, deepcopy
from functools import reduce


class MultiMaskArray(numpy.lib.mixins.NDArrayOperatorsMixin):

    def __init__(self, values, masks=None):
        self._values = values
        self._masks = masks if masks is not None else {}

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def values(self):
        return self._values

    @property
    def masks(self):
        return self._masks

    def _flat_mask(self):
        return reduce(lambda x, y: np.logical_or(x, y), self._masks.values())

    def __repr__(self):
        return f"{self.__class__.__name__}(values={self._values}, masks={self._masks})"

    def __array__(self, dtype=None):
        if self._masks:
            return np.ma.MaskedArray(self._values, mask=self._flat_mask())
        else:
            return self._values

    def __copy__(self):
        """Copy behaving like NumPy copy, i.e., making a copy of the buffers."""
        return self.__class__(copy(self._values), deepcopy(self._masks))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            arrays = []
            masks = {}
            for x in inputs:
                if isinstance(x, MultiMaskArray):
                    arrays.append(x._values)
                    for key, mask in x.masks.items():
                        if key in masks:
                            masks[key] = np.logical_or(masks[key], mask)
                        else:
                            masks[key] = mask
                elif isinstance(x, np.ndarray):
                    arrays.append(x)
                else:
                    return NotImplemented
            return self.__class__(ufunc(*arrays, **kwargs), masks=masks)
        else:
            return NotImplemented

    def __array_function__(self):
        pass
