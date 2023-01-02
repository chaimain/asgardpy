"""
Classes containing the Dataset Reduction config parameters for the high-level
interface
"""

from enum import Enum
from pathlib import Path
from typing import List

from asgardpy.data.base import BaseConfig, TimeRangeConfig, AngleType

__all__ = [
    "ReductionTypeEnum",
    "RequiredHDUEnum",
    "ObservationsConfig",
    "BackgroundRegionFinderMethodEnum",
    "ReflectedRegionFinderConfig",
    "WobbleRegionsFinderConfig",
    "BackgroundMethodEnum",
    "SafeMaskMethodsEnum",
    "MapSelectionEnum",
    "BackgroundConfig",
    "SafeMaskConfig",
]


class ReductionTypeEnum(str, Enum):
    spectrum = "1d"
    cube = "3d"


class RequiredHDUEnum(str, Enum):
    aeff = "aeff"
    bkg = "bkg"
    edisp = "edisp"
    psf = "psf"
    rad_max = "rad_max"
    point_like = "point-like"
    full_enclosure = "full-enclosure"


class ObservationsConfig(BaseConfig):
    obs_ids: List[int] = []
    obs_file: Path = None
    obs_time: TimeRangeConfig = TimeRangeConfig()
    required_irfs: List[RequiredHDUEnum] = ["aeff"]


class BackgroundMethodEnum(str, Enum):
    reflected = "reflected"
    fov = "fov_background"
    ring = "ring"


class BackgroundRegionFinderMethodEnum(str, Enum):
    reflected = "reflected"
    wobble = "wobble"


class ReflectedRegionFinderConfig(BaseConfig):
    angle_increment: AngleType = None
    min_distance: AngleType = None
    min_distance_input: AngleType = None
    max_region_number: int = 10000
    binsz: AngleType = None


class WobbleRegionsFinderConfig(BaseConfig):
    n_off_regions: int = 1
    binsz: AngleType = None


class SafeMaskMethodsEnum(str, Enum):
    aeff_default = "aeff-default"
    aeff_max = "aeff-max"
    edisp_bias = "edisp-bias"
    offset_max = "offset-max"
    bkg_peak = "bkg-peak"
    custom_mask = "custom-mask"


class MapSelectionEnum(str, Enum):
    counts = "counts"
    exposure = "exposure"
    background = "background"
    psf = "psf"
    edisp = "edisp"


# Dataset Reduction Makers config
class BackgroundConfig(BaseConfig):
    method: BackgroundMethodEnum = None
    region_finder_method: BackgroundRegionFinderMethodEnum = None
    parameters: dict = {}
    exclusion: dict = {}


class SafeMaskConfig(BaseConfig):
    methods: List[SafeMaskMethodsEnum] = []
    parameters: dict = {}
