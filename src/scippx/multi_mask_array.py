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
        return f"{self.__class__.__name__}({self._values}, masks={self._masks})"

    def __getitem__(self, key):
        return self.__class__(self._values[key],
                              {name: mask[key]
                               for name, mask in self.masks.items()})

    def __setitem__(self, key, value):
        self._values[key] = value
        for name, mask in value.masks.items():
            self.masks[name][key] = mask

    def __array__(self, dtype=None):
        # TODO apply masks?
        return self._values.__array__()

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
                else:
                    arrays.append(x)
            if (out := kwargs.get('out')) is not None:
                kwargs['out'] = tuple(
                    [v.values if isinstance(v, MultiMaskArray) else v for v in out])
            return self.__class__(ufunc(*arrays, **kwargs), masks=masks)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        # TODO handle more args, this works for concatenate and broadcast_arrays
        def values(arg):
            if isinstance(arg, MultiMaskArray):
                return arg.values
            if isinstance(arg, list):
                return [values(x) for x in arg]
            return arg
        arrays = tuple([values(x) for x in args])
        values = func(*arrays, **kwargs)
        masks = {}
        for name in self.masks:
            masks[name] = func([x.masks[name] for x in args[0]], **kwargs)
        return self.__class__(values, masks)

    def __getattr__(self, item):
        try:
            return getattr(self.values, item)
        except AttributeError:
            raise AttributeError("Neither MultiMaskArray object nor its data ({}) "
                                 "has attribute '{}'".format(self.values, item))
