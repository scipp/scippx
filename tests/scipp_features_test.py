import numpy as np
import scippx.scipp as sc
import pint
import pytest


def test_linspace():
    x = sc.linspace('x', 0, 1, 3, units='m')
    assert x.data.units == sc.Unit('m')
    assert x.dims == ('x', )
    assert x.coords == {}


def test_array():
    da = sc.array(dims=('xx', ), values=np.arange(4))
    assert da.data.units == ''
    assert len(da.masks) == 0
    assert type(da.data) == sc.Quantity


def test_array_with_units():
    da = sc.array(dims=('xx', ), values=np.arange(4), units='m')
    assert da.data.units == sc.Unit('m')


def test_array_with_dimension_coord():
    x = sc.linspace('x', 0.1, 0.2, 4, units='s')
    da = sc.array(dims=('x', ), values=np.arange(4), coords={'x': x})
    # Fails because da.coords['x2'] has same coords as da:
    # assert da.coords['x'].equals(x)
    assert da.scipp.coords['x'].equals(x)
    assert da.scipp.coords['x'].data.units == sc.Unit('s')


def test_array_with_non_dimension_coord():
    x2 = sc.linspace('x', 0.1, 0.2, 4, units='s')
    da = sc.array(dims=('x', ), values=np.arange(4), coords={'x2': x2})
    # Fails because da.coords['x2'] has same coords as da:
    # assert da.coords['x2'].equals(x2)
    assert da.scipp.coords['x2'].equals(x2)


def test_array_with_dimension_coord_label_based_lookup():
    x = sc.linspace('x', 0.1, 0.4, 4, units=None)
    da = sc.array(dims=('x', ), values=np.arange(4), coords={'x': x})
    assert da.loc[0.2].equals(da[1])


def test_array_with_dimension_coord_label_based_lookup_raiss_if_unit_not_specified():
    x = sc.linspace('x', 0.1, 0.4, 4, units='s')
    da = sc.array(dims=('x', ), values=np.arange(4), coords={'x': x})
    with pytest.raises(pint.errors.DimensionalityError):
        da.sel(x=0.2)
    with pytest.raises(pint.errors.DimensionalityError):
        da.loc[0.2]


def test_array_scipp_quantity_lookup():
    x = sc.linspace('x', 0.1, 0.4, 4, units='s')
    da = sc.array(dims=('x', ), values=np.arange(4), coords={'x': x})
    sel = da.scipp[('x', 0.2 * sc.Unit('s'))]
    assert sel.equals(da[1:2])


def test_array_quantity_lookup():
    x = sc.linspace('x', 0.1, 0.4, 4, units='s')
    da = sc.array(dims=('x', ), values=np.arange(4), coords={'x': x})
    # sc.Unit is pint Unit with force_ndarray_like=True so this works
    sel = da.sel(x=0.2 * sc.Unit('s'))
    assert sel.equals(da[1])
    sel = da.loc[{'x': 0.2 * sc.Unit('s')}]
    assert sel.equals(da[1])


def test_coords_support_inplace_modification_and_get_reflected_in_index():
    x = sc.linspace('x', 0.1, 0.4, 4, units='s')
    da = sc.array(dims=('x', ), values=np.arange(4), coords={'x': x})
    assert da.sel(x=0.2 * sc.Unit('s')).equals(da[1])
    da.coords['x'] *= 2
    assert da.sel(x=0.2 * sc.Unit('s')).equals(da[0])


def test_as_edges():
    x = sc.linspace('x', 0.1, 0.4, 4, units='s')
    edges = sc.as_edges(x)
    assert edges.dims == ('x', )
    assert edges.data.units == sc.Unit('s')


def test_array_with_edges():
    x = sc.as_edges(sc.linspace('x', 0.1, 0.5, 5, units='s'))
    da = sc.array(dims=('x', ), values=np.arange(4), coords={'x': x})
    # TODO Figure out how to make ArrayIndex work with BinEdgeArray, in particular
    # how to handle comparisons or things such as np.nonzero and np.argmax
