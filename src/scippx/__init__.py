# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

# flake8: noqa
import importlib.metadata
try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import accessors
from .bin_edge_array import BinEdgeArray
from .multi_mask_array import MultiMaskArray
from .uncertain_array import UncertainArray
from . import scipp
