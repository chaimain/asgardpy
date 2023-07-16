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
    get_chi2_pval,
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
    "BrokenPowerLaw2SpectralModel",
    "Dataset1DConfig",
    "Dataset1DGeneration",
    "Dataset3DConfig",
    "Dataset3DGeneration",
    "Datasets1DAnalysisStep",
    "Datasets3DAnalysisStep",
    "ExpCutoffLogParabolaSpectralModel",
    "FitAnalysisStep",
    "FitConfig",
    "FluxPointsAnalysisStep",
    "FluxPointsConfig",
    "Target",
    "apply_selection_mask_to_models",
    "check_model_preference_aic",
    "check_model_preference_lrt",
    "config_to_dict",
    "create_gal_diffuse_skymodel",
    "create_iso_diffuse_skymodel",
    "create_source_skymodel",
    "get_chi2_pval",
    "set_models",
    "xml_spatial_model_to_gammapy",
    "xml_spectral_model_to_gammapy_params",
]
