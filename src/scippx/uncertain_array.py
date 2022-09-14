# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import numpy as np
import numpy.lib.mixins
from copy import copy, deepcopy


class UncertainArray(numpy.lib.mixins.NDArrayOperatorsMixin):

    def __init__(self, values, variances):
        self._values = values
        self._variances = variances

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
                                      variances=np.sum(variances, axis=0))
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __array_function__(self):
        pass
