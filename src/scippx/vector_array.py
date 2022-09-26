from typing import List
import numpy as np
import numpy.lib.mixins
from .array_attr import ArrayAttrMixin

HANDLED_FUNCTIONS = {}


def implements(numpy_function):
    """Register an __array_function__ implementation for VectorArray objects."""

    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


@implements(np.empty_like)
def empty_like(prototype, dtype=None, order='K', subok=True, shape=None):
    nfield = len(prototype.fields)
    shape = (shape if isinstance(shape, tuple) else (shape, )) + (nfield, )
    values = np.empty_like(prototype.values,
                           dtype=dtype,
                           order=order,
                           subok=subok,
                           shape=shape)
    return VectorArray(values, prototype._field_names)


@implements(np.concatenate)
def concatenate(args, axis=0, out=None, dtype=None, casting="same_kind"):
    assert out is None
    values = np.concatenate(tuple(args.values for arg in args),
                            axis=axis,
                            dtype=stype,
                            casting=casting)
    return VectorArray(values, args[0]._field_names)


@implements(np.dot)
def dot(a, b):
    # Note difference to np.dot: our "dtype" enables a simpler and better definition
    return np.einsum('...i,...i->...', a.values, b.values)


@implements(np.amax)
def amax(a, axis=None):
    if axis is not None and len(axis) and max(axis) >= a.ndim:
        # Avoid comuting over internal axes
        raise ValueError("Axis index too large")
    return VectorArray(np.amax(a.values, axis=axis), a._field_names)


@implements(np.sum)
def sum(a, axis=None):
    if axis is not None and len(axis) and max(axis) >= a.ndim:
        # Avoid comuting over internal axes
        raise ValueError("Axis index too large")
    return VectorArray(np.sum(a.values, axis=axis), a._field_names)


class Fields:

    def __init__(self, obj, wrap=None, unwrap=None):
        self._obj = obj
        self._wrap = (lambda x: x) if wrap is None else wrap
        self._unwrap = (lambda x: x) if unwrap is None else unwrap

    def __len__(self):
        return len(self._obj._field_names)

    def __getitem__(self, key):
        field = self._obj.values[..., self._obj._field_names.index(key)]
        return self._wrap(field)

    def __setitem__(self, key, value):
        self._obj.values[..., self._obj._field_names.index(key)] = self._unwrap(value)


class VectorArray(numpy.lib.mixins.NDArrayOperatorsMixin, ArrayAttrMixin):
    """Array of vectors with named components (fields)."""

    def __init__(self, values: np.ndarray, field_names: List[str]):
        # Array-of-structure implementation
        assert values.shape[-1] == len(field_names)
        self._values = values
        self._field_names = field_names

    def __repr__(self):
        return f"{self.__class__.__name__}(field_names={self.field_names}, shape={self.shape}, values={self.values})"

    @property
    def field_names(self):
        return self._field_names

    @property
    def shape(self):
        return self._values.shape[:-1]

    @property
    def ndim(self):
        return self._values.ndim - 1

    def __getitem__(self, index):
        return VectorArray(self._values[index], self.field_names)

    def _require_same_field_names(self, other):
        if other.field_names != self.field_names:
            raise ValueError(
                f"Incompatible VectorArray(field_names={other.field_names}), "
                f"expected {self.field_names}.")

    def __setitem__(self, key, value):
        self._require_same_field_names(value)
        self.values[key] = value.values

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
        if method == '__call__':
            arrays = []
            vector_count = 0
            for x in inputs:
                if isinstance(x, VectorArray):
                    vector_count += 1
                    if x.field_names != self.field_names:
                        raise ValueError(f"Incompatible field names {x.field_names}")
                    arrays.append(x._values)
                else:
                    arrays.append(x)
            if ufunc == np.multiply and vector_count > 1:
                raise ValueError("Vectors cannot be multiplied. Did you mean dot()?")
            if (out := kwargs.get('out')) is not None:
                kwargs['out'] = tuple(
                    [v._values if isinstance(v, VectorArray) else v for v in out])
            return self.__class__(ufunc(*arrays, **kwargs),
                                  field_names=self.field_names)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle VectorArray objects
        if not all(issubclass(t, VectorArray) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def _rewrap_content(self, content):
        return self.__class__(content, self.field_names)

    def _unwrap_content(self, obj):
        self._require_same_field_names(obj)
        return obj.values

    def __array_property__(self, name, wrap, unwrap):
        if name == 'fields':
            return Fields(self, wrap=wrap, unwrap=unwrap)
        return self._forward_array_getattr_to_content(name, wrap, unwrap)
