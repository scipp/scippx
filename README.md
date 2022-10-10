[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

# Scippx

Experimentation with implementation of Scipp features based on Xarray or duck arrays in general.
This mainly demonstrates a possible model of interaction between multiple duck-array implementations in a "stack" of duck array layers.

## Duck arrays

This prototype currently uses or provides the following "duck arrays".
A central goal is to evaluate if the current duck-array mechanism is capable of combining all these in a coherent and user-friendly manner:

Duck array | scope
---|---
`dask.array.Array`| array for parallel computing
`pint.Quantity`| array with`units`
`xarray.Variable`| array with `dims`
`xarray.DataArray`| array with `coords` dict
`MultiMaskArray`| array with `masks` dict
`BinEdgeArray`| array wrapping N+1 edges
`VectorArray`| array of vectors

### Demonstrated concepts

The tests in `test/array_property_test.py` demonstrate the main idea.
Central parts of the implementation are found in `src/scippx/array_property.py` and `src/scippx/array_attr.py`.
The following concepts are demonstrated:

- Mechanism for letting duck-array implementation expose properties or methods on all wrapping levels.
  The mechanism ensures that important properties from higher levels in the duck array stack are not dropped.
  For example, accessing the `magnitude` property of a `pint.Quantity` layer does *not* lose information about dimension labels of the wrapping `xr.Variable`.
- Mechanism for modifying or removing duck-array layers.
  As above this keep (and validates) information from wrapping layers, such as dimension labels.
- Use with `dask`, without involvement of wrapping layers, i.e., duck arrays do not require special handling to support wrapping dask arrays.
