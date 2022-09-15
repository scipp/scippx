import numpy as np
import xarray as xr
import pint
import pytest


def test_binary_ops_are_not_associative_given_non_range_slices():
    da = xr.DataArray(dims=['x'], data=[1, 2, 3], coords={'x': ('x', [1, 2, 3])})
    # Turns x into coord without index
    a = da[0]
    b = da[1]
    c = da[2]
    a_bc = a + (b + c)
    ab_c = (a + b) + c
    assert not a_bc.equals(ab_c)
    assert a_bc.coords['x'].equals(a.coords['x'])
    assert ab_c.coords['x'].equals(c.coords['x'])


def test_binary_ops_are_not_associative_given_coords_without_indexes():
    da = xr.DataArray(dims=['x'], data=[1, 2, 3, 4], coords={'x2': ('x', [1, 2, 3, 4])})
    a = da[0:2]
    b = da[1:3]
    c = da[2:4]
    a_bc = a + (b + c)
    ab_c = (a + b) + c
    assert not a_bc.equals(ab_c)
    assert a_bc.coords['x2'].equals(a.coords['x2'])
    assert ab_c.coords['x2'].equals(c.coords['x2'])


def test_binary_op_with_mismatching_index_returns_empty():
    da = xr.DataArray(dims=['x'], data=[1, 2, 3], coords={'x': [1, 2, 3]})
    assert len(da[0:1] + da[1:2]) == 0


def test_binary_op_with_mismatching_length_and_index_returns_empty():
    da = xr.DataArray(dims=['x'], data=[1, 2, 3], coords={'x': [1, 2, 3]})
    assert len(da[0:1] + da[1:3]) == 0


def test_binary_op_without_index_and_matching_length_returns_with_dropped_coord():
    da = xr.DataArray(dims=['x'], data=[1, 2, 3], coords={'x2': ('x', [1, 2, 3])})
    result = da[0:1] + da[1:2]
    assert 'x2' not in result.coords


def test_binary_op_without_index_and_mismatching_length_raises():
    da = xr.DataArray(dims=['x'], data=[1, 2, 3], coords={'x2': ('x', [1, 2, 3])})
    with pytest.raises(ValueError):
        da[0:1] + da[1:3]


@pytest.mark.filterwarnings(
    "ignore:elementwise comparison failed; this will raise an error in the future.:DeprecationWarning"
)
def test_equals_of_index_coord_ignores_units_since_it_only_compares_values():
    a = xr.Variable(dims='x', data=np.arange(4))
    b = xr.Variable(dims='x', data=pint.Quantity(np.arange(4), 'm'))
    assert not b.equals(a)  # ok
    assert not a.equals(b)  # ok
    da = xr.DataArray(dims='x', data=a, coords={'x': a})
    assert not b.equals(da.coords['x'].variable)  # ok
    assert not xr.DataArray(da.coords['x'].variable).equals(b)  # ok
    assert da.coords['x'].variable.equals(b)  # bad
