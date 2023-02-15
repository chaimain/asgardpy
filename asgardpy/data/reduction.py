"""
Classes containing the Dataset Reduction config parameters for the high-level
interface.
"""

from enum import Enum
from typing import List

from astropy import units as u

from asgardpy.data.base import AngleType, BaseConfig, TimeIntervalsConfig
from asgardpy.data.geom import SkyCoordConfig

__all__ = [
    "ReductionTypeEnum",
    "RequiredHDUEnum",
    "ObservationsConfig",
    "BackgroundRegionFinderMethodEnum",
    "ReflectedRegionFinderConfig",
    "WobbleRegionsFinderConfig",
    "BackgroundMethodEnum",
    "ExclusionRegionsConfig",
    "RegionsConfig",
    "SafeMaskMethodsEnum",
    "MapSelectionEnum",
    "BackgroundConfig",
    "SafeMaskConfig",
]


# Basic Components to define the various Dataset Reduction Maker Config
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
    obs_file: str = "."
    obs_time: TimeIntervalsConfig = TimeIntervalsConfig()
    required_irfs: List[RequiredHDUEnum] = [RequiredHDUEnum.aeff]


class BackgroundMethodEnum(str, Enum):
    reflected = "reflected"
    fov = "fov_background"
    ring = "ring"


class BackgroundRegionFinderMethodEnum(str, Enum):
    reflected = "reflected"
    wobble = "wobble"


class ReflectedRegionFinderConfig(BaseConfig):
    angle_increment: AngleType = 0.01 * u.deg
    min_distance: AngleType = 0.1 * u.deg
    min_distance_input: AngleType = 0.1 * u.deg
    max_region_number: int = 10000
    binsz: AngleType = 0.05 * u.deg


class WobbleRegionsFinderConfig(BaseConfig):
    n_off_regions: int = 1
    binsz: AngleType = 0.05 * u.deg


class RegionsConfig(BaseConfig):
    type: str = ""
    name: str = ""
    position: SkyCoordConfig = SkyCoordConfig()
    parameters: dict = {}


class ExclusionRegionsConfig(BaseConfig):
    target_source: bool = True
    regions: List[RegionsConfig] = []


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
    method: BackgroundMethodEnum = BackgroundMethodEnum.reflected
    region_finder_method: BackgroundRegionFinderMethodEnum = BackgroundRegionFinderMethodEnum.wobble
    parameters: dict = {}
    exclusion: ExclusionRegionsConfig = ExclusionRegionsConfig()


class SafeMaskConfig(BaseConfig):
    methods: List[SafeMaskMethodsEnum] = []
    parameters: dict = {}
