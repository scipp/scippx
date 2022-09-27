[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

# Scippx

Experimentation with implementation of Scipp features based on Xarray or duck arrays in general.

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