# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import numpy as np
import numpy.lib.mixins
from copy import copy, deepcopy
from .array_attr import ArrayAttrMixin, rewrap_result


def empty_like(prototype, dtype=None, order='K', subok=True, shape=None):
    assert len(shape) == 1
    if shape[0] == 0:
        shape = shape
    else:
        shape = (shape[0] + 1)
    edges = np.empty_like(prototype.edges,
                          dtype=dtype,
                          order=order,
                          subok=subok,
                          shape=shape)
    return BinEdgeArray(edges)


def concatenate(args, axis=0, out=None, dtype=None, casting="same_kind"):
    assert out is None
    first, *rest = args
    # TODO check compatible left and right
    args = (first.edges, ) + (x.right for x in rest)
    return BinEdgeArray(np.concatenate(args, axis=axis, dtype=dtype))


def amax(a, axis=None):
    if axis is not None and axis != (a.ndim - 1):  # TODO check tuple
        return BinEdgeArray(np.amax(a.edges, axis=axis))
    else:  # edge-axis is removed, do not return BinEdgeArray but underlying array
        return np.amax(a.edges, axis=None)


class BinEdgeArray(numpy.lib.mixins.NDArrayOperatorsMixin, ArrayAttrMixin):

    def __init__(self, edges):
        assert edges.ndim == 1  # TODO
        # TODO check increasing
        self._values = edges

    @property
    def shape(self):
        # TODO >1d
        return (self.edges.shape[0] - 1, )

    @property
    def dtype(self):
        return self.edges.dtype

    @property
    def ndim(self):
        return self.edges.ndim

    @property
    def edges(self):
        return self._values

    @property
    def left(self):
        return self._values[:-1]

    def center(self):
        return 0.5 * (self.right - self.left)

    @property
    def right(self):
        return self._values[1:]

    def __len__(self):
        return len(self._values) - 1

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, edges={self._values})"

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = key[0]  # TODO
        if isinstance(key, int):
            # TODO shape? return class Interval?
            return self.__class__(self._values[key:key + 2])
        else:
            return self.__class__(self._values[slice(key.start, key.stop + 1)])

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        if isinstance(key, slice):
            start = key.start
            stop = key.stop
            key = slice(start, stop + 1)
        self.edges[key] = value.edges

    def __array__(self, dtype=None):
        # TODO Should this return midpoints? Or self.left?
        return self._values.__array__()

    def __copy__(self):
        """Copy behaving like NumPy copy, i.e., making a copy of the buffer."""
        return self.__class__(copy(self._values))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            arrays = []
            for x in inputs:
                if isinstance(x, BinEdgeArray):
                    arrays.append(x._values)
                else:
                    arrays.append(x)
            if (out := kwargs.get('out')) is not None:
                kwargs['out'] = tuple(
                    [v._values if isinstance(v, BinEdgeArray) else v for v in out])
            return self.__class__(ufunc(*arrays, **kwargs))
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        if func == np.concatenate:
            return concatenate(*args, **kwargs)
        if func == np.empty_like:
            return empty_like(*args, **kwargs)
        if func == np.amax:
            return amax(*args, **kwargs)
        if func == np.sum:
            raise RuntimeError("Summing BinEdgeArray is not possible. "
                               "Try summing the `centers()` or `edges`.")
        return NotImplemented

    def _rewrap_content(self, content):
        return self.__class__(content)

    def _unwrap_content(self, obj):
        # TODO Do we need to verify shape?
        return obj.edges

    def __array_property__(self, name, wrap, unwrap):
        if name == 'left':
            return wrap(self.left)
        if name == 'right':
            return wrap(self.right)
        if name == 'center':
            return rewrap_result(wrap)(self.center)
        return self._forward_array_getattr_to_content(name, wrap, unwrap)
