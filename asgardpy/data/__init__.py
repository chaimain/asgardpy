from gammapy.utils.registry import Registry

from asgardpy.data.base import AnalysisStep, AnalysisStepBase
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
    "AnalysisStepBase",
    "AnalysisStep",
    "Dataset1DInfoConfig",
    "Dataset1DBaseConfig",
    "Dataset1DConfig",
    "Dataset1DDataSelectionAnalysisStep",
    "Dataset1DObservationsAnalysisStep",
    "Dataset1DDatasetsAnalysisStep" "Dataset3DInfoConfig",
    "Dataset3DBaseConfig",
    "Dataset3DConfig",
    "Dataset3DDataSelectionAnalysisStep",
    "Dataset3DObservationsAnalysisStep",
    "Dataset3DDatasetsAnalysisStep",
]
