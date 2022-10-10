# Adapted from https://notebooksharing.space/view/48ad86aed90f7588c9a475be6747528d87f975cb3317e5bd94265ffaa5a2478f#displayOptions=
# (by benbovy)
# Needs xarray/main

from __future__ import annotations

from typing import Any, TYPE_CHECKING, Mapping, Hashable, Iterable, Sequence

import numpy as np
import xarray as xr

from xarray.core.indexes import Index, PandasIndex, IndexVars, is_scalar
from xarray.core.indexing import IndexSelResult
from xarray.core import nputils
from xarray.core.variable import Variable, IndexVariable

#if TYPE_CHECKING:
from xarray.core.types import JoinOptions, T_Index


class ArrayIndex(Index):
    """Numpy-like array index.
    
    Lightweight, inefficient index as a basic wrapper around
    its coordinate array data.
    
    This index is suited for cases where index build overhead
    is an issue and where only basic indexing operations are
    needed (i.e., strict alignment, data selection in rare occasions).
    
    """
    array: np.ndarray
    dim: Hashable
    name: Hashable

    # cause AttributeError with `da + da` example below?????
    #__slots__ = ("array", "dim", "name")

    def __init__(self, array, dim, name):
        if array.ndim > 1:
            raise ValueError("ArrayIndex only accepts 1-dimensional arrays")

        self.array = array
        self.dim = dim
        self.name = name

    @classmethod
    def from_variables(cls: type[T_Index], variables: Mapping[Any, Variable], options):
        if len(variables) != 1:
            raise ValueError(
                f"PandasIndex only accepts one variable, found {len(variables)} variables"
            )

        name, var = next(iter(variables.items()))

        # TODO: use `var.data` instead? (allow lazy/duck arrays)
        return cls(var.data, var.dims[0], name)

    @classmethod
    def concat(
        cls: type[T_Index],
        indexes: Sequence[T_Index],
        dim: Hashable,
        positions: Iterable[Iterable[int]] = None,
    ) -> T_Index:
        if not indexes:
            return cls(np.array([]), dim, dim)

        if not all(idx.dim == dim for idx in indexes):
            dims = ",".join({f"{idx.dim!r}" for idx in indexes})
            raise ValueError(f"Cannot concatenate along dimension {dim!r} indexes with "
                             f"dimensions: {dims}")

        arrays = [idx.array for idx in indexes]
        new_array = np.concatenate(arrays)

        if positions is not None:
            indices = nputils.inverse_permutation(np.concatenate(positions))
            new_array = new_array.take(indices)

        return cls(new_array, dim, indexes[0].name)

    def create_variables(self,
                         variables: Mapping[Any, Variable] | None = None) -> IndexVars:

        #
        # TODO: implementation is needed so that the corresponding
        # coordinate is indexed properly with Dataset.isel.
        # Ideally this shouldn't be needed, though.
        #

        if variables is not None and self.name in variables:
            var = variables[self.name]
            attrs = var.attrs
            encoding = var.encoding
        else:
            attrs = None
            encoding = None

        var = Variable(self.dim, self.array, attrs=attrs, encoding=encoding)
        return {self.name: var}

    def isel(
        self: T_Index, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> T_Index | PandasIndex | None:
        indxr = indexers[self.dim]

        if isinstance(indxr, Variable):
            if indxr.dims != (self.dim, ):
                # can't preserve a index if result has new dimensions
                return None
            else:
                indxr = indxr.data
        if not isinstance(indxr, slice) and is_scalar(indxr):
            # scalar indexer: drop index
            return None

        return type(self)(self.array[indxr], self.dim, self.name)

    def sel(self, labels: dict[Any, Any], **kwargs) -> IndexSelResult:
        assert len(labels) == 1
        _, label = next(iter(labels.items()))

        if isinstance(label, slice):
            # TODO: what exactly do we want to do here?
            _ = self.array[0] + label.start  # Duck compatibility check, e.g., unit
            _ = self.array[0] + label.stop  # Duck compatibility check, e.g., unit
            start = np.argmax(self.array == label.start)
            stop = np.argmax(self.array == label.stop)
            indexer = slice(start, stop)
        elif is_scalar(label):
            _ = self.array[0] + label  # Duck compatibility check, e.g., unit
            indexer = np.argmax(self.array == label)
        else:
            # TODO: other label types we want to support (n-d array-like, etc.)
            raise ValueError(f"label {label} not supported by ArrayIndex")

        return IndexSelResult({self.dim: indexer})

    def equals(self: T_Index, other: T_Index) -> bool:
        return np.array_equal(self.array, other.array)

    def roll(self: T_Index, shifts: Mapping[Any, int]) -> T_Index:
        shift = shifts[self.dim]

        return type(self)(np.roll(self.array, shift), self.dim, self.name)
