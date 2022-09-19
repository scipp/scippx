# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import numpy as np
import numpy.lib.mixins
from copy import copy, deepcopy
from functools import reduce


class ListArray(numpy.lib.mixins.NDArrayOperatorsMixin):

    def __init__(self, *, starts, stops, content, axis=0):
        self._starts: np.ndarray = starts
        self._stops: np.ndarray = stops
        self._axis: int = axis  # axis of content referenced by starts and stops
        self._content = content  # array-like

    @property
    def shape(self):
        return self._starts.shape

    @property
    def ndim(self):
        return self._starts.ndim

    def __repr__(self):
        return f"{self.__class__.__name__}(starts={self._starts},...)"

    def __getitem__(self, key):
        return self.__class__(starts=self._starts[key],
                              stops=self._stops[key],
                              axis=self._axis,
                              content=self._content)

    def __array__(self, dtype=None):
        raise NotImplementedError()
        # TODO apply masks?
        return self._values.__array__()

    def __copy__(self):
        """Copy behaving like NumPy copy, i.e., making a copy of the buffers."""
        return self.__class__(starts=copy(self._starts),
                              stops=copy(self._stops),
                              axis=self._axis,
                              content=deepcopy(self._content))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError()
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
        raise NotImplementedError()
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
