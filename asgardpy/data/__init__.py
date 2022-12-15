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
from asgardpy.data.dataset_1d import (
    Dataset1DDataSelectionAnalysisStep,
    Dataset1DDatasetsAnalysisStep,
    Dataset1DObservationsAnalysisStep,
)
from asgardpy.data.dataset_3d import (
    Dataset3DDataSelectionAnalysisStep,
    Dataset3DDatasetsAnalysisStep,
    Dataset3DObservationsAnalysisStep,
)
from asgardpy.data.geom import (
    EnergyAxesConfig,
    EnergyAxisConfig,
    FinalFrameConfig,
    GeomConfig,
    OnRegion,
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
from asgardpy.data.target import TargetModel, TargetSource

ANALYSIS_STEP_REGISTRY = Registry(
    [
        Dataset1DDataSelectionAnalysisStep,
        Dataset1DObservationsAnalysisStep,
        Dataset1DDatasetsAnalysisStep,
        Dataset3DDataSelectionAnalysisStep,
        Dataset3DObservationsAnalysisStep,
        Dataset3DDatasetsAnalysisStep,
        # ExcessMapAnalysisStep,
        # FitAnalysisStep,
        # FluxPointsAnalysisStep,
        # LightCurveAnalysisStep,
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
    "TargetSource",
    "TargetModel",
    "SpatialCircleConfig",
    "SpatialPointConfig",
    "OnRegion",
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
    "Dataset1DDataSelectionAnalysisStep",
    "Dataset1DObservationsAnalysisStep",
    "Dataset1DDatasetsAnalysisStep",
    "Dataset3DDataSelectionAnalysisStep",
    "Dataset3DObservationsAnalysisStep",
    "Dataset3DDatasetsAnalysisStep",
]
