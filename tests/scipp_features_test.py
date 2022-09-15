import numpy as np
import scippx.scipp as sc
import pint


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
