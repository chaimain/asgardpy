from gammapy.utils.registry import Registry

from asgardpy.data.base import (
    AnalysisStep,
    AnalysisStepBase,
    AngleType,
    BaseConfig,
    EnergyRangeConfig,
    EnergyType,
    FrameEnum,
    TimeRangeConfig,
    TimeType,
)
from asgardpy.data.dataset_1d import Dataset1DGeneration, Datasets1DAnalysisStep
from asgardpy.data.dataset_3d import Dataset3DGeneration, Datasets3DAnalysisStep
from asgardpy.data.dl4 import (
    ExcessMapConfig,
    FitAnalysisStep,
    FitConfig,
    FluxPointsAnalysisStep,
    FluxPointsConfig,
    LightCurveAnalysisStep,
    LightCurveConfig,
)
from asgardpy.data.geom import (
    EnergyAxesConfig,
    EnergyAxisConfig,
    FinalFrameConfig,
    GeomConfig,
    SelectionConfig,
    SkyCoordConfig,
    SpatialCircleConfig,
    SpatialPointConfig,
    WcsConfig,
)
from asgardpy.data.reduction import (
    BackgroundConfig,
    BackgroundMethodEnum,
    MapSelectionEnum,
    ReductionTypeEnum,
    RequiredHDUEnum,
    SafeMaskConfig,
    SafeMaskMethodsEnum,
)
from asgardpy.data.target import Target, config_to_dict, set_models

ANALYSIS_STEP_REGISTRY = Registry(
    [
        Datasets1DAnalysisStep,
        Datasets3DAnalysisStep,
        # ExcessMapAnalysisStep,
        FitAnalysisStep,
        FluxPointsAnalysisStep,
        LightCurveAnalysisStep,
    ]
)

__all__ = [
    "AngleType",
    "EnergyType",
    "TimeType",
    "FrameEnum",
    "BaseConfig",
    "AnalysisStepBase",
    "AnalysisStep",
    "TimeRangeConfig",
    "EnergyRangeConfig",
    "Target",
    "set_models",
    "config_to_dict",
    "SpatialCircleConfig",
    "SpatialPointConfig",
    "EnergyAxisConfig",
    "EnergyAxesConfig",
    "SelectionConfig",
    "FinalFrameConfig",
    "SkyCoordConfig",
    "WcsConfig",
    "GeomConfig",
    "ReductionTypeEnum",
    "RequiredHDUEnum",
    "BackgroundMethodEnum",
    "SafeMaskMethodsEnum",
    "MapSelectionEnum",
    "BackgroundConfig",
    "SafeMaskConfig",
    "Dataset1DGeneration",
    "Datasets1DAnalysisStep",
    "Dataset3DGeneration",
    "Datasets3DAnalysisStep",
    "FluxPointsConfig",
    "LightCurveConfig",
    "FitConfig",
    "ExcessMapConfig",
    "FitAnalysisStep",
    "FluxPointsAnalysisStep",
    "LightCurveAnalysisStep",
]
