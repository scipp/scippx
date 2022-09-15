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
