"""
asgardpy - Gammapy-based pipeline for easy joint analysis of different gamma-ray datasets

Licensed under `Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
See `License <https://github.com/chaimain/asgardpy/blob/main/LICENSE>`_.
"""
from . import analysis, base, config, data, io, stats
from .version import VERSION, VERSION_SHORT

__all__ = [
    "analysis",
    "base",
    "config",
    "data",
    "io",
    "stats",
    "VERSION",
    "VERSION_SHORT",
]
