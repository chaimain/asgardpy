from gammapy.utils.registry import Registry

from asgardpy.data.dataset_1d import (
    Dataset1DConfig,
    Dataset1DGeneration,
    Datasets1DAnalysisStep,
)
from asgardpy.data.dataset_3d import (
    Dataset3DConfig,
    Dataset3DGeneration,
    Datasets3DAnalysisStep,
)
from asgardpy.data.dl4 import (
    FitAnalysisStep,
    FitConfig,
    FluxPointsAnalysisStep,
    FluxPointsConfig,
)
from asgardpy.data.target import (
    BrokenPowerLaw2SpectralModel,
    ExpCutoffLogParabolaSpectralModel,
    Target,
    apply_selection_mask_to_models,
    check_model_preference_aic,
    check_model_preference_lrt,
    config_to_dict,
    create_gal_diffuse_skymodel,
    create_iso_diffuse_skymodel,
    create_source_skymodel,
    set_models,
    xml_spatial_model_to_gammapy,
    xml_spectral_model_to_gammapy_params,
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
    "Target",
    "BrokenPowerLaw2SpectralModel",
    "ExpCutoffLogParabolaSpectralModel",
    "set_models",
    "apply_selection_mask_to_models",
    "config_to_dict",
    "xml_spectral_model_to_gammapy_params",
    "xml_spatial_model_to_gammapy",
    "check_model_preference_aic",
    "check_model_preference_lrt",
    "create_source_skymodel",
    "create_iso_diffuse_skymodel",
    "create_gal_diffuse_skymodel",
    "SpatialCircleConfig",
    "SpatialPointConfig",
    "Dataset1DConfig",
    "Dataset1DGeneration",
    "Datasets1DAnalysisStep",
    "Dataset3DConfig",
    "Dataset3DGeneration",
    "Datasets3DAnalysisStep",
    "FluxPointsConfig",
    "FitConfig",
    "FitAnalysisStep",
    "FluxPointsAnalysisStep",
]
