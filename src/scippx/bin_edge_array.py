# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import numpy as np
import numpy.lib.mixins
from copy import copy, deepcopy
from .array_attr import ArrayAttrMixin, rewrap_result

HANDLED_FUNCTIONS = {}


def implements(numpy_function):
    """Register an __array_function__ implementation for VectorArray objects."""

    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


@implements(np.empty_like)
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


@implements(np.concatenate)
def concatenate(args, axis=0, out=None, dtype=None, casting="same_kind"):
    assert out is None
    first, *rest = args
    for left, right in zip(args[:-1], args[1:]):
        if left.edges[-1] != right.edges[0]:
            raise ValueError("Incompatible value at edge bounds")
    args = (first.edges, ) + tuple(x.right for x in rest)
    return BinEdgeArray(np.concatenate(args, axis=axis, dtype=dtype))


@implements(np.amax)
def amax(a, axis=None):
    if axis is not None and axis != (a.ndim - 1):  # TODO check tuple
        return BinEdgeArray(np.amax(a.edges, axis=axis))
    else:  # edge-axis is removed, do not return BinEdgeArray but underlying array
        return np.amax(a.edges, axis=None)


class BinEdgeArray(numpy.lib.mixins.NDArrayOperatorsMixin, ArrayAttrMixin):
    """
    Array of values stored on the bin/cell bounaries.

    Wraps an underlying array of values of shape=(..., N+1) as a duck array
    of shape=(..., N).
    """

    def __init__(self, edges):
        assert edges.ndim >= 1
        assert edges.shape[-1] != 1  # must be 0 or >=2
        self._edges = edges

    @property
    def shape(self):
        if self._edges.shape[-1] == 0:
            return self._edges.shape
        return self._edges.shape[:-1] + (self._edges.shape[-1] - 1, )

    @property
    def dtype(self):
        return self.edges.dtype

    @property
    def ndim(self):
        return self.edges.ndim

    @property
    def edges(self):
        return self._edges

    @property
    def left(self):
        return self.edges[..., :-1]

    def center(self):
        return 0.5 * (self.right - self.left)

    @property
    def right(self):
        return self._edges[..., 1:]

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, edges={self._edges})"

    def __getitem__(self, key):
        if self.ndim != 1:
            raise NotImplementedError("Slicing for non-1-D edges not implemented")
        if isinstance(key, tuple):
            key = key[0]  # TODO
        if isinstance(key, int):
            # TODO shape? return class Interval?
            return self.__class__(self._edges[key:key + 2])
        else:
            start, stop, stride = key.indices(self.shape[-1])
            if stop > start:
                stop += 1
            return self.__class__(self._edges[slice(start, stop, stride)])

    def __setitem__(self, key, value):
        if self.ndim != 1:
            raise NotImplementedError("Slicing for non-1-D edges not implemented")
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        if isinstance(key, slice):
            start = key.start
            stop = None if key.stop is None else key.stop + 1
            key = slice(start, stop)
        self.edges[key] = value.edges

    def __array__(self, dtype=None):
        # TODO Should this raise?
        return self.edges.__array__()

    def __copy__(self):
        """Copy behaving like NumPy copy, i.e., making a copy of the buffer."""
        return self.__class__(copy(self._edges))

    def _rewrap_content(self, content):
        return self.__class__(content)

    def _unwrap_content(self, obj):
        # TODO Do we need to verify shape?
        if hasattr(obj, 'shape'):
            return obj.edges
        return obj  # scalar such as int or float

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            arrays = [self._unwrap_content(x) for x in inputs]
            if (out := kwargs.get('out')) is not None:
                kwargs['out'] = tuple(self._unwrap_content(v) for v in out)
            return self.__class__(ufunc(*arrays, **kwargs))
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func == np.sum:
            raise RuntimeError("Summing BinEdgeArray is not possible. "
                               "Try summing the `centers()` or `edges`.")
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, BinEdgeArray) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __array_property__(self, name, wrap, unwrap):
        if name == 'left':
            return wrap(self.left)
        if name == 'right':
            return wrap(self.right)
        if name == 'center':
            return rewrap_result(wrap)(self.center)
        return self._forward_array_getattr_to_content(name, wrap, unwrap)
