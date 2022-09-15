import xarray as xr
import pytest


def test_binary_ops_are_not_associative_given_coords_without_index():
    da = xr.DataArray(dims=['x'], data=[1, 2, 3], coords={'x2': ('x', [1, 2, 3])})
    a = da[0]
    b = da[1]
    c = da[2]
    a_bc = a + (b + c)
    ab_c = (a + b) + c
    assert not a_bc.equals(ab_c)


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
