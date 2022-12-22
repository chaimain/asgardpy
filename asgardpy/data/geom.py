"""
Classes containing the Geometry config parameters for the high-level interface
"""

from enum import Enum

from asgardpy.data.base import AngleType, BaseConfig, EnergyType, FrameEnum

__all__ = [
    "SpatialCircleConfig",
    "SpatialPointConfig",
    "EnergyAxisConfig",
    "EnergyAxesConfig",
    "SelectionConfig",
    "FinalFrameConfig",
    "SkyCoordConfig",
    "WcsConfig",
    "GeomConfig",
]


class SpatialCircleConfig(BaseConfig):
    frame: FrameEnum = None
    lon: AngleType = None
    lat: AngleType = None
    radius: AngleType = None


class SpatialPointConfig(BaseConfig):
    frame: FrameEnum = None
    lon: AngleType = None
    lat: AngleType = None


class EnergyAxisConfig(BaseConfig):
    min: EnergyType = None
    max: EnergyType = None
    nbins: int = None


class EnergyAxesConfig(BaseConfig):
    energy: EnergyAxisConfig = EnergyAxisConfig(min="1 TeV", max="10 TeV", nbins=5)
    energy_true: EnergyAxisConfig = EnergyAxisConfig(min="0.5 TeV", max="20 TeV", nbins=16)


class SelectionConfig(BaseConfig):
    offset_max: AngleType = "2.5 deg"


class FinalFrameConfig(BaseConfig):
    width: AngleType = "5 deg"
    height: AngleType = "5 deg"


class SkyCoordConfig(BaseConfig):
    frame: FrameEnum = None
    lon: AngleType = None
    lat: AngleType = None


class WcsConfig(BaseConfig):
    skydir: SkyCoordConfig = SkyCoordConfig()
    binsize: AngleType = "0.02 deg"
    final_frame: FinalFrameConfig = FinalFrameConfig()
    binsize_irf: AngleType = "0.2 deg"


class GeomConfig(BaseConfig):
    wcs: WcsConfig = WcsConfig()
    selection: SelectionConfig = SelectionConfig()
    axes: EnergyAxesConfig = EnergyAxesConfig()
