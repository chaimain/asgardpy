"""
Classes containing the DL4 products config parameters for the high-level interface
"""
from asgardpy.data.base import AngleType, BaseConfig, EnergyRangeConfig, TimeRangeConfig
from asgardpy.data.geom import EnergyAxisConfig

__all__ = [
    "FluxPointsConfig",
    "LightCurveConfig",
    "FitConfig",
    "ExcessMapConfig",
]


class FluxPointsConfig(BaseConfig):
    energy: EnergyAxisConfig = EnergyAxisConfig()
    source: str = "source"
    parameters: dict = {"selection_optional": "all"}


class LightCurveConfig(BaseConfig):
    time_intervals: TimeRangeConfig = TimeRangeConfig()
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()
    source: str = "source"
    parameters: dict = {"selection_optional": "all"}


class FitConfig(BaseConfig):
    fit_range: EnergyRangeConfig = EnergyRangeConfig()


class ExcessMapConfig(BaseConfig):
    correlation_radius: AngleType = "0.1 deg"
    parameters: dict = {}
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()
