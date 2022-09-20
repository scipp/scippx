from typing import List
import numpy as np
import numpy.lib.mixins

class Fields:

    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, key):
        return self._obj[self._field_names.index(key)]


class VectorArray(numpy.lib.mixins.NDArrayOperatorsMixin):

    def __init__(self, values: np.ndarray, field_names: List[str]):
        # Assuming a structure-of-array implementation
        assert len(values) == len(field_names)
        self._values = values
        self._field_names = field_names

    @property
    def shape(self):
        return self._values.shape[1:]

    def __getitem__(self, index):
        return VectorArray(self._values[:, index], self._components)

    @property
    def fields(self):
        return Fields(self)

    @property
    def values(self):
        return self._values

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc == np.multiply:
            raise ValueError("Vectors cannot be multiplied. Did you mean dot()?")
        if method == '__call__':
            arrays = []
            for x in inputs:
                if isinstance(x, VectorArray):
                    if x._field_names != self._field_names:
                        raise ValueError(f"Incompatible field names {x._field_names}")
                    arrays.append(x._values)
                else:
                    arrays.append(x)
            if (out := kwargs.get('out')) is not None:
                kwargs['out'] = tuple(
                    [v._values if isinstance(v, VectorArray) else v for v in out])
            return self.__class__(ufunc(*arrays, **kwargs), field_names=self._field_names)
        else:
            return NotImplemented

    def __array_function__(self):
        """TODO"""
