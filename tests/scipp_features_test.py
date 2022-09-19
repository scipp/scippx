import numpy as np
import scippx.scipp as sc
import pint
import pytest


def test_linspace():
    x = sc.linspace('x', 0, 1, 3, units='m')
    assert x.data.units == pint.Unit('m')
    assert x.dims == ('x', )
    assert x.coords == {}


def test_array():
    da = sc.array(dims=('xx', ), values=np.arange(4))
    assert da.data.units == ''
    assert len(da.masks) == 0
    assert type(da.data) == pint.Quantity


def test_array_with_units():
    da = sc.array(dims=('xx', ), values=np.arange(4), units='m')
    assert da.data.units == pint.Unit('m')


def test_array_with_dimension_coord():
    x = sc.linspace('x', 0.1, 0.2, 4, units='s')
    da = sc.array(dims=('x', ), values=np.arange(4), coords={'x': x})
    # Fails because da.coords['x2'] has same coords as da:
    # assert da.coords['x'].equals(x)
    assert da.scipp.coords['x'].equals(x)
    assert da.scipp.coords['x'].data.units == pint.Unit('s')


def test_array_with_non_dimension_coord():
    x2 = sc.linspace('x', 0.1, 0.2, 4, units='s')
    da = sc.array(dims=('x', ), values=np.arange(4), coords={'x2': x2})
    # Fails because da.coords['x2'] has same coords as da:
    # assert da.coords['x2'].equals(x2)
    assert da.scipp.coords['x2'].equals(x2)


def test_array_with_dimension_coord_label_based_lookup_raises_since_no_index():
    x = sc.linspace('x', 0.1, 0.2, 4, units='s')
    da = sc.array(dims=('x', ), values=np.arange(4), coords={'x': x})
    with pytest.raises(KeyError):
        da.loc[0.1]


def test_array_scipp_quantity_lookup():
    x = sc.linspace('x', 0.1, 0.4, 4, units='s')
    da = sc.array(dims=('x', ), values=np.arange(4), coords={'x': x})
    sel = da.scipp[('x', 0.2 * pint.Unit('s'))]
    assert sel.equals(da[1:2])


def test_array_quantity_lookup():
    x = sc.linspace('x', 0.1, 0.4, 4, units='s')
    da = sc.array(dims=('x', ), values=np.arange(4), coords={'x': x})
    sel = da.sel(x=np.array(0.2) * pint.Unit('s'))
    assert sel.equals(da[1])
    sel = da.loc[{'x':np.array(0.2) * pint.Unit('s')}]
    assert sel.equals(da[1])
