"""
Classes containing the Geometry config parameters for the high-level interface.
"""

from enum import Enum
from typing import List

from astropy import units as u
from gammapy.maps import MapAxis

from asgardpy.data.base import AngleType, BaseConfig, EnergyType, FrameEnum

__all__ = [
    "SpatialCircleConfig",
    "SpatialPointConfig",
    "EnergyAxisConfig",
    "MapAxesConfig",
    "EnergyEdgesCustomConfig",
    "SelectionConfig",
    "MapFrameShapeConfig",
    "SkyCoordConfig",
    "ProjectionEnum",
    "WcsConfig",
    "GeomConfig",
    "get_energy_axis",
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
    per_decade: bool = True


class EnergyEdgesCustomConfig(BaseConfig):
    edges: List[float] = []
    unit: str = "TeV"


class MapAxesConfig(BaseConfig):
    name: str = "energy"
    axis: EnergyAxisConfig = EnergyAxisConfig()
    axis_custom: EnergyEdgesCustomConfig = EnergyEdgesCustomConfig()


class SelectionConfig(BaseConfig):
    offset_max: AngleType = 2.5 * u.deg


class MapFrameShapeConfig(BaseConfig):
    width: AngleType = 5 * u.deg
    height: AngleType = 5 * u.deg


class SkyCoordConfig(BaseConfig):
    frame: FrameEnum = FrameEnum.icrs
    lon: AngleType = 0 * u.deg
    lat: AngleType = 0 * u.deg


class ProjectionEnum(str, Enum):
    tan = "TAN"
    car = "CAR"


class WcsConfig(BaseConfig):
    skydir: SkyCoordConfig = SkyCoordConfig()
    binsize: AngleType = 0.1 * u.deg
    proj: ProjectionEnum = ProjectionEnum.tan
    map_frame_shape: MapFrameShapeConfig = MapFrameShapeConfig()
    binsize_irf: AngleType = 0.2 * u.deg


class GeomConfig(BaseConfig):
    wcs: WcsConfig = WcsConfig()
    selection: SelectionConfig = SelectionConfig()
    axes: List[MapAxesConfig] = [MapAxesConfig()]
    from_events_file: bool = True


def get_energy_axis(axes_config, only_edges=False, custom_range=False):
    """
    Read from a MapAxesConfig section to return the required energy axis/range.
    """
    energy_axis = axes_config.axis
    energy_axis_custom = axes_config.axis_custom

    if custom_range:
        energy_range = energy_axis_custom.edges * u.Unit(energy_axis_custom.unit)
    else:
        energy_range = MapAxis.from_energy_bounds(
            energy_min=u.Quantity(energy_axis.min),
            energy_max=u.Quantity(energy_axis.max),
            nbin=int(energy_axis.nbins),
            per_decade=energy_axis.per_decade,
            name=energy_axis.name,
        )
        if only_edges:
            energy_range = energy_range.edges

    return energy_range
