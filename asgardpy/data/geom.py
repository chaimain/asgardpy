"""
Classes containing the Geometry config parameters for the high-level interface.
"""

from typing import List

from astropy import units as u

from asgardpy.data.base import AngleType, BaseConfig, EnergyType, FrameEnum

__all__ = [
    "SpatialCircleConfig",
    "SpatialPointConfig",
    "EnergyAxisConfig",
    "EnergyAxesConfig",
    "EnergyEdgesCustomConfig",
    "SelectionConfig",
    "FinalFrameConfig",
    "SkyCoordConfig",
    "WcsConfig",
    "GeomConfig",
]


# Basic Components to define the main GeomConfig
class SpatialCircleConfig(BaseConfig):
    frame: FrameEnum = FrameEnum.icrs
    lon: AngleType = 0 * u.deg
    lat: AngleType = 0 * u.deg
    radius: AngleType = 0.1 * u.rad


class SpatialPointConfig(BaseConfig):
    frame: FrameEnum = FrameEnum.icrs
    lon: AngleType = 0 * u.deg
    lat: AngleType = 0 * u.deg


class EnergyAxisConfig(BaseConfig):
    min: EnergyType = 1 * u.GeV
    max: EnergyType = 1 * u.TeV
    nbins: int = 5


class EnergyEdgesCustomConfig(BaseConfig):
    edges: List[EnergyType] = []


class EnergyAxesConfig(BaseConfig):
    energy: EnergyAxisConfig = EnergyAxisConfig(min=1 * u.TeV, max=10 * u.TeV, nbins=5)
    energy_true: EnergyAxisConfig = EnergyAxisConfig(min=0.5 * u.TeV, max=20 * u.TeV, nbins=16)


class SelectionConfig(BaseConfig):
    offset_max: AngleType = 2.5 * u.deg


class FinalFrameConfig(BaseConfig):
    width: AngleType = 5 * u.deg
    height: AngleType = 5 * u.deg


class SkyCoordConfig(BaseConfig):
    frame: FrameEnum = FrameEnum.icrs
    lon: AngleType = 0 * u.deg
    lat: AngleType = 0 * u.deg


class WcsConfig(BaseConfig):
    skydir: SkyCoordConfig = SkyCoordConfig()
    binsize: AngleType = 0.02 * u.deg
    final_frame: FinalFrameConfig = FinalFrameConfig()
    binsize_irf: AngleType = 0.2 * u.deg


class GeomConfig(BaseConfig):
    wcs: WcsConfig = WcsConfig()
    selection: SelectionConfig = SelectionConfig()
    axes: EnergyAxesConfig = EnergyAxesConfig()
