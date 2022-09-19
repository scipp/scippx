import numpy as np
import scippx as sx


def test_create():
    starts = np.array([0, 2])
    stops = np.array([2, 3])
    content = np.array([0.1, 0.2, 0.3])
    la = sx.ListArray(starts=starts, stops=stops, content=content)
    assert la.shape == (2, )
