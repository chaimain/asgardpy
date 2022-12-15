"""classes containing the analysis steps supported by the high level interface"""

import abc
from enum import Enum
from gammapy.utils.scripts import make_path #, make_name
import logging
from asgardpy.config.generator import (
    AngleType,
    EnergyAxisConfig,
    FrameEnum,
    SkyCoordConfig,
    TimeRangeConfig,
)

__all__ = [
    "AnalysisStepBase",
    "AnalysisStep",
    "ReductionTypeEnum",
    "FrameEnum",
    "RequiredHDUEnum",
    "BackgroundMethodEnum",
    "SafeMaskMethodsEnum",
    "MapSelectionEnum",
    "SpatialCircleConfig",
    "SpatialPointConfig",
    "OnRegion",
    "BackgroundConfig",
    "SafeMaskConfig",
    "EnergyAxesConfig",
    "SelectionConfig",
    "FinalFrameConfig",
    "WcsConfig",
    "GeomConfig",
    "ObservationsConfig"
]


class AnalysisStepBase(abc.ABC):
    tag = "analysis-step"

    def __init__(self, config, log=None, name=None, overwrite=True):
        self.config = config
        self.overwrite = overwrite
        # self._name = make_name(name)

        if log is None:
            log = logging.getLogger(__name__)
            self.log = log

    @property
    def name(self):
        return self._name

    def run(self): #, data
        #self.data = data
        #self.products = self.data
        self._run()

    @abc.abstractmethod
    def _run(self):
        pass


class AnalysisStep:
    "Create one of the analysis step class listed in the registry"

    @staticmethod
    def create(tag, config, **kwargs):
        from . import ANALYSIS_STEP_REGISTRY

        cls = ANALYSIS_STEP_REGISTRY.get_cls(tag)
        return cls(config, **kwargs)


class ReductionTypeEnum(str, Enum):
    spectrum = "1d"
    cube = "3d"


class FrameEnum(str, Enum):
    icrs = "icrs"
    galactic = "galactic"


class RequiredHDUEnum(str, Enum):
    events = "events"
    gti = "gti"
    aeff = "aeff"
    bkg = "bkg"
    edisp = "edisp"
    psf = "psf"
    rad_max = "rad_max"
    point_like = "point-like"
    full_enclosure = "full-enclosure"


class BackgroundMethodEnum(str, Enum):
    reflected = "reflected"
    fov = "fov_background"
    ring = "ring"
    region_finder = "region-finder"


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


# Define the different ON regions
class SpatialCircleConfig(BaseConfig):
    frame: FrameEnum = None
    lon: AngleType = None
    lat: AngleType = None
    radius: AngleType = None


class SpatialPointConfig(BaseConfig):
    frame: FrameEnum = None
    lon: AngleType = None
    lat: AngleType = None


class OnRegion(str, Enum):
    point_region = SpatialPointConfig()
    circle_region = SpatialCircleConfig()


# Dataset Reduction Makers config
class BackgroundConfig(BaseConfig):
    method: BackgroundMethodEnum = None
    exclusion: dict = {}
    parameters: dict = {} # Only for 1D dataset?


class SafeMaskConfig(BaseConfig):
    methods: List[SafeMaskMethodsEnum] = [SafeMaskMethodsEnum.aeff_default]
    parameters: dict = {}


# Dataset geom config
class EnergyAxesConfig(BaseConfig):
    energy: EnergyAxisConfig = EnergyAxisConfig(
        min="1 TeV", max="10 TeV", nbins=5
    )
    energy_true: EnergyAxisConfig = EnergyAxisConfig(
        min="0.5 TeV", max="20 TeV", nbins=16
    )


class SelectionConfig(BaseConfig):
    offset_max: AngleType = "2.5 deg"


class FinalFrameConfig(BaseConfig):
    width: AngleType = "5 deg"
    height: AngleType = "5 deg"


class WcsConfig(BaseConfig):
    skydir: SkyCoordConfig = SkyCoordConfig()
    binsize: AngleType = "0.02 deg"
    final_frame: FinalFrameConfig = FinalFrameConfig()
    binsize_irf: AngleType = "0.2 deg"


class GeomConfig(BaseConfig):
    wcs: WcsConfig = WcsConfig()
    selection: SelectionConfig = SelectionConfig()
    axes: EnergyAxesConfig = EnergyAxesConfig()


# More info for general observations
class ObservationsConfig(BaseConfig):
    obs_ids: List[int] = []
    obs_file: Path = None
    obs_time: TimeRangeConfig = TimeRangeConfig()
    required_irf: List[RequiredHDUEnum] = ["aeff", "edisp", "psf", "bkg"]
