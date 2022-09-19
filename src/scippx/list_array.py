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
        assert axis == 0  # TODO
        self._axis: int = axis  # axis of content referenced by starts and stops
        self._content = content  # array-like

    @property
    def shape(self):
        return self._starts.shape

    @property
    def ndim(self):
        return self._starts.ndim

    @property
    def sizes(self):
        return self._stops - self._starts

    @property
    def values(self):
        if self.shape == ():
            return self._content[slice(self._starts, self._stops)]
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}(starts={self._starts},...)"

    def __getitem__(self, key):
        return self.__class__(starts=np.asarray(self._starts[key]),
                              stops=np.asarray(self._stops[key]),
                              axis=self._axis,
                              content=self._content)

    def __setitem__(self, key, other):
        if not isinstance(other, ListArray):
            raise NotImplementedError()
        if not np.array_equal(self[key].sizes, other.sizes):
            raise ValueError("mismatching bin sizes")
        sel = self[key]
        for start, stop, ostart, ostop in zip(np.ravel(sel._starts),
                                              np.ravel(sel._stops),
                                              np.ravel(other._starts),
                                              np.ravel(other._stops)):
            self._content[slice(start, stop)] = other._content[slice(ostart, ostop)]

    def __array__(self, dtype=None):
        raise NotImplementedError()

    def __copy__(self):
        """Copy behaving like NumPy copy, i.e., making a copy of the buffers."""
        sizes = self.sizes
        stops = np.reshape(np.cumsum(np.ravel(sizes)), sizes.shape)
        starts = stops - sizes
        size = np.sum(sizes)
        # TODO See uptream efforts for API to create correct duck array
        # TODO Support non-1D content
        content = np.empty_like(self._content, shape=(size, ))
        out = self.__class__(starts=starts,
                             stops=stops,
                             axis=self._axis,
                             content=content)
        out[...] = self
        return out

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError()

    def __array_function__(self, func, types, args, kwargs):
        raise NotImplementedError()
