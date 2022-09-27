# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import numpy as np
import numpy.lib.mixins
from copy import copy, deepcopy
from functools import reduce
from .array_attr import ArrayAttrMixin

HANDLED_FUNCTIONS = {}


def implements(numpy_function):
    """Register an __array_function__ implementation for MultiMaskArray objects."""

    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


@implements(np.concatenate)
def concatenate(args, axis=0, dtype=None, casting="same_kind"):
    data = np.concatenate(tuple(arg.data for arg in args),
                          axis=axis,
                          dtype=dtype,
                          casting=casting)
    masks = {}
    for mask in args[0].masks:
        masks[mask] = np.concatenate(tuple(arg.masks[mask] for arg in args),
                                     axis=axis,
                                     dtype=dtype,
                                     casting=casting)
    return MultiMaskArray(data, masks)


@implements(np.amax)
def amax(a, axis=None):
    # TODO Handle multi-dim case and apply only relevant masks
    mask = a._flat_mask()
    ma = np.ma.array(a.data, mask=a._flat_mask())
    return MultiMaskArray(np.amax(ma))


class Masks:

    def __init__(self, obj, wrap=None, unwrap=None):
        self._obj = obj
        self._wrap = (lambda x: x) if wrap is None else wrap
        self._unwrap = (lambda x: x) if unwrap is None else unwrap

    def __len__(self):
        return len(self._obj._masks)

    def __getitem__(self, name):
        mask = self._obj._masks[name]
        return self._wrap(mask)

    def __setitem__(self, name, mask):
        if mask.shape != self._obj.shape:
            raise ValueError(f"Incompatible shape={mask.shape} for mask '{name}'")
        self._obj._masks[name] = self._unwrap(mask)

    def __contains__(self, name):
        return name in self._obj._masks

    def __iter__(self):
        yield from self._obj._masks


class MultiMaskArray(numpy.lib.mixins.NDArrayOperatorsMixin, ArrayAttrMixin):

    def __init__(self, data, masks=None):
        self._data = data
        self._masks = {}
        for name, mask in {} if masks is None else masks.items():
            self.masks[name] = mask

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def data(self):
        return self._data

    @property
    def masks(self):
        return Masks(self)

    def _flat_mask(self):
        return reduce(lambda x, y: np.logical_or(x, y), self._masks.values())

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data}, masks={self._masks})"

    def __getitem__(self, key):
        return self.__class__(self._data[key],
                              {name: mask[key]
                               for name, mask in self._masks.items()})

    def __setitem__(self, key, value):
        self._data[key] = value._data
        # TODO what does this do if masks dict empty?
        for name, mask in value._masks.items():
            self._masks[name][key] = mask

    def __array__(self, dtype=None):
        raise RuntimeError("Cannot convert MultiMaskArray to ndarray")

    def __copy__(self):
        """Copy behaving like NumPy copy, i.e., making a copy of the buffers."""
        return self.__class__(copy(self._data), deepcopy(self._masks))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            arrays = []
            masks = {}
            for x in inputs:
                if isinstance(x, MultiMaskArray):
                    arrays.append(x._data)
                    for key, mask in x._masks.items():
                        if key in masks:
                            masks[key] = np.logical_or(masks[key], mask)
                        else:
                            masks[key] = mask
                else:
                    arrays.append(x)
            if (out := kwargs.get('out')) is not None:
                kwargs['out'] = tuple(
                    [v.data if isinstance(v, MultiMaskArray) else v for v in out])
            return self.__class__(ufunc(*arrays, **kwargs), masks=masks)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, MultiMaskArray) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def _rewrap_content(self, content):
        data, *masks = content
        return self.__class__(data, dict(zip(self._masks, masks)))

    def _unwrap_content(self, obj):
        # Unlike other examples such as VectorArray or Quantity, our content consists
        # of *multiple* arrays.
        return (obj.data, ) + tuple(obj._masks.values())

    def __array_getattr__(self, name, wrap, unwrap):
        if name == 'unmasked':
            return wrap(self.data)
        if name == 'masks':
            return Masks(self, wrap=wrap, unwrap=unwrap)
        # This illustrates a potential wider problem: If our duck array
        # may wrap *multiple* other arrays we may need to operate on all of
        # them. For example, we may imagine a RecordArray implemented as a
        # dict of arrays, each of which may be a dask array.
        # The wrapped arrays may also be heterogenous, so some may require
        # forwarding while some may not. ArrayAttrMixin tries some magic to deal with
        # this.
        return self._forward_array_getattr_to_content(name, wrap, unwrap)
