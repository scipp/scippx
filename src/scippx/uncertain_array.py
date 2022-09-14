# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import numpy as np
import numpy.lib.mixins
from copy import copy, deepcopy

try:
    from dask import array as dask_array
    from dask.base import compute, persist, visualize, DaskMethodsMixin, replace_name_in_key
except ImportError:
    compute, persist, visualize = None, None, None
    DaskMethodsMixin = None
    dask_array = None


def check_dask_array(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if isinstance(self._magnitude, dask_array.Array):
            return f(self, *args, **kwargs)
        else:
            msg = "Method {} only implemented for objects of {}, not {}".format(
                f.__name__, dask_array.Array, self._magnitude.__class__
            )
            raise AttributeError(msg)

    return wrapper


class UncertainArray(numpy.lib.mixins.NDArrayOperatorsMixin, DaskMethodsMixin):

    def __init__(self, values, variances):
        self._values = values
        self._variances = variances
        print(f'init {len(self)}')

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
    def variances(self):
        return self._variances

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return f"{self.__class__.__name__}(values={self.values}, variances={self.variances})"

    def __getitem__(self, key):
        return self.__class__(self.values[key], self.variances[key])

    def __setitem__(self, key, other):
        self.values[key] = other.values
        self.variances[key] = other.variances

    def __array__(self, dtype=None):
        return self.values.__array__()

    def __copy__(self):
        """Copy behaving like NumPy copy, i.e., making a copy of the buffers."""
        return self.__class__(copy(self.values), copy(self.variances))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            if ufunc in [np.add, np.subtract]:
                # TODO many things wrong here
                values = []
                variances = []
                for x in inputs:
                    if isinstance(x, UncertainArray):
                        values.append(x.values)
                        variances.append(x.variances)
                    else:
                        values.append(x)
                if (out := kwargs.get('out')) is not None:
                    kwargs['out'] = tuple([
                        v.values if isinstance(v, UncertainArray) else v for v in out
                    ])
                return self.__class__(ufunc(*values, **kwargs),
                                      variances=np.add(*variances, **kwargs))
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __array_function__(self):
        pass

    def __dask_graph__(self):
        print('__dask_graph__')
        graph = {}
        graph.update(self.values.__dask_graph__())
        graph.update(self.variances.__dask_graph__())
        return graph

    def __dask_keys__(self):
        print('__dask_keys__')
        return [self.values.__dask_keys__(), self.variances.__dask_keys__()]

    @property
    def __dask_optimize__(self):
        return dask_array.Array.__dask_optimize__

    @property
    def __dask_scheduler__(self):
        return dask_array.Array.__dask_scheduler__

    def __dask_postcompute__(self):
        print('__dask_postcompute__')
        func, val_args = self.values.__dask_postcompute__()
        func, var_args = self.variances.__dask_postcompute__()
        return self._dask_finalize, (func, val_args, var_args)

    def __dask_postpersist__(self):
        func, val_args = self.values.__dask_postpersist__()
        func, var_args = self.variances.__dask_postpersist__()
        return self._dask_finalize, (func, val_args, var_args)

    @staticmethod
    def _dask_finalize(results, func, val_args, var_args):
        values = func(results[0], *val_args)
        variances = func(results[1], *var_args)
        return UncertainArray(values, variances)

    #def __dask_postcompute__(self):
    #    #func, val_args = self.values.__dask_postcompute__()
    #    #func, var_args = self.variances.__dask_postcompute__()
    #    def make(args):
    #        return self.__class__(*args)
    #    return make, ()

    #def __dask_postpersist__(self):
    #    return self.__class__, 
    #    # We need to return a callable with the signature
    #    # rebuild(dsk, *extra_args, rename: Mapping[str, str] = None)
    #    return UncertainArray._rebuild, (self._keys,)

    #@staticmethod
    #def _rebuild(dsk, keys, *, rename=None):
    #    if rename is not None:
    #        keys = [replace_name_in_key(key, rename) for key in keys]
    #    return self.__class__(dsk, keys)

    #@staticmethod
    #def _dask_finalize(results, func, args, units):
    #    values = func(results, *args)
    #    return Quantity(values, units)
