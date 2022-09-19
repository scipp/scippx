import numpy as np
import xarray as xr
import pint
import scippx as sx
import pytest


@pytest.fixture()
def array_with_array_index():
    x = xr.Variable(dims='x', data=[1.1, 2.2, 3.3, 4.4])
    data = xr.Variable(dims='x', data=np.array([2, 3, 4, 5]))
    da = xr.DataArray(data, coords={'x': x})
    return da.drop_indexes('x').set_xindex('x', sx.ArrayIndex)


def test_sel(array_with_array_index):
    da = array_with_array_index
    assert 'x' in da.coords
    assert 'x' in da.xindexes
    da_sel = da.sel(x=2.2)
    assert da_sel.shape == ()
    assert da_sel.values == 3


def test_binary_op_with_matching_coord_works(array_with_array_index):
    da = array_with_array_index
    result = da + da
    assert result.variable.equals(da.variable + da.variable)


def test_raises_on_coord_mismatch(array_with_array_index):
    da = array_with_array_index
    # TODO Implement `join` to raise nice exception
    with pytest.raises(NotImplementedError):
        da[1:] + da[:-1]


@pytest.fixture()
def array_with_quantity_array_index():
    x = xr.Variable(dims='x', data=pint.Quantity([1.1, 2.2, 3.3, 4.4], 'm'))
    ix = sx.ArrayIndex(x, 'x', 'x')
    data = xr.Variable(dims='x', data=np.array([2, 3, 4, 5]))
    da = xr.DataArray(data, coords={'x': x}, indexes={}, fastpath=True)
    return da.set_xindex('x', sx.ArrayIndex)


def test_sel_with_pint_quantity_raises_DimensionalityError_on_unit_mismatch(
        array_with_quantity_array_index):
    da = array_with_quantity_array_index
    assert da.coords['x'].data.units == pint.Unit('m')
    with pytest.raises(pint.errors.DimensionalityError):
        da.sel(x=2.2)


def test_sel_with_pint_quantity(array_with_quantity_array_index):
    da = array_with_quantity_array_index
    assert da.coords['x'].data.units == pint.Unit('m')
    da_sel = da.sel(x=np.array(2.2) * pint.Unit('m'))
    # This is not working currenty, as this is not identified as a scalar
    # da_sel = da.sel(x=2.2 * pint.Unit('m'))
    # da_sel = da.sel(x=pint.Quantity(2.2, 'm'))
