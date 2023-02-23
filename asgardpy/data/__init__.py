from gammapy.utils.registry import Registry

from asgardpy.data.base import (
    AnalysisStep,
    AnalysisStepBase,
    AngleType,
    BaseConfig,
    EnergyRangeConfig,
    EnergyType,
    FrameEnum,
    TimeFormatEnum,
    TimeIntervalsConfig,
    TimeRangeConfig,
    TimeType,
)
from asgardpy.data.dataset_1d import Dataset1DGeneration, Datasets1DAnalysisStep
from asgardpy.data.dataset_3d import Dataset3DGeneration, Datasets3DAnalysisStep
from asgardpy.data.dl4 import (
    FitAnalysisStep,
    FitConfig,
    FluxPointsAnalysisStep,
    FluxPointsConfig,
)
from asgardpy.data.geom import (
    EnergyAxesConfig,
    EnergyAxisConfig,
    EnergyEdgesCustomConfig,
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
    ExclusionRegionsConfig,
    MapSelectionEnum,
    ReductionTypeEnum,
    RegionsConfig,
    RequiredHDUEnum,
    SafeMaskConfig,
    SafeMaskMethodsEnum,
)
from asgardpy.data.target import (
    ExpCutoffLogParabolaSpectralModel,
    Target,
    config_to_dict,
    create_gal_diffuse_skymodel,
    create_iso_diffuse_skymodel,
    create_source_skymodel,
    set_models,
    xml_to_gammapy_model_params,
)

ANALYSIS_STEP_REGISTRY = Registry(
    [
        Datasets1DAnalysisStep,
        Datasets3DAnalysisStep,
        FitAnalysisStep,
        FluxPointsAnalysisStep,
    ]
)

__all__ = [
    "AngleType",
    "EnergyType",
    "TimeType",
    "FrameEnum",
    "TimeFormatEnum",
    "BaseConfig",
    "AnalysisStepBase",
    "AnalysisStep",
    "TimeRangeConfig",
    "TimeIntervalsConfig",
    "EnergyRangeConfig",
    "Target",
    "ExpCutoffLogParabolaSpectralModel",
    "set_models",
    "config_to_dict",
    "xml_to_gammapy_model_params",
    "create_source_skymodel",
    "create_iso_diffuse_skymodel",
    "create_gal_diffuse_skymodel",
    "SpatialCircleConfig",
    "SpatialPointConfig",
    "EnergyAxisConfig",
    "EnergyAxesConfig",
    "EnergyEdgesCustomConfig",
    "ExclusionRegionsConfig",
    "SelectionConfig",
    "FinalFrameConfig",
    "SkyCoordConfig",
    "WcsConfig",
    "GeomConfig",
    "ReductionTypeEnum",
    "RequiredHDUEnum",
    "RegionsConfig",
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
    "FitConfig",
    "FitAnalysisStep",
    "FluxPointsAnalysisStep",
]
