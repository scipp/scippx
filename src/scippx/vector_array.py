from typing import List
import numpy as np
import numpy.lib.mixins


class Fields:

    def __init__(self, obj, wrap = None):
        self._obj = obj
        self._wrap = wrap

    def __getitem__(self, key):
        field = self._obj.values[self._obj._field_names.index(key)]
        return field if self._wrap is None else self._wrap(field)


class VectorArray(numpy.lib.mixins.NDArrayOperatorsMixin):

    def __init__(self, values: np.ndarray, field_names: List[str]):
        # Assuming a structure-of-array implementation
        assert len(values) == len(field_names)
        self._values = values
        self._field_names = field_names

    @property
    def shape(self):
        return self._values.shape[1:]

    @property
    def ndim(self):
        return self._values.ndim - 1

    def __getitem__(self, index):
        return VectorArray(self._values[:, index], self._field_names)

    @property
    def dtype(self):
        # TODO this does not sound right
        return self.values.dtype

    @property
    def fields(self):
        return Fields(self)

    @property
    def values(self):
        return self._values

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc == np.multiply:
            # TODO We should allow scaling with a scalar
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
            return self.__class__(ufunc(*arrays, **kwargs),
                                  field_names=self._field_names)
        else:
            return NotImplemented

    def __array_function__(self):
        """TODO"""

    def __array_property__(self, name, wrap):
        if name == 'fields':
            return Fields(self, wrap)
        if hasattr(self._values, '__array_property__'):
            return self._values.__array_property__(
                name, wrap=lambda x: wrap(self.__class__(x, self._field_names)))
        raise AttributeError(f"{self.__class__} object has no attribute '{name}'")
