import numpy as np
import xarray as xr
import scippx as sx
from pint import Quantity
import pytest


@pytest.fixture
def masked_1d_array():
    x = xr.Variable(dims=['x'], data=Quantity(np.arange(5), 'm'))
    data = sx.MultiMaskArray(np.arange(5),
                             masks={'mask': np.ones(shape=(5, ), dtype=bool)})
    data = xr.Variable(dims=['x'], data=data)
    return xr.DataArray(data=data, coords={'x': x, 'x2': x}, indexes={}, fastpath=True)


def test_mask_access(masked_1d_array):
    da = masked_1d_array
    assert da.masks['mask'].dims == ('x', )


def test_set_value_of_mask_slice(masked_1d_array):
    da = masked_1d_array
    assert da.masks['mask'].values[1] == True
    da[{'x': 1}].masks['mask'][()] = False
    assert da.masks['mask'].values[1] == False
