"""
Classes containing the Target config parameters for the high-level interface and
also the functions involving Models generation and assignment to datasets.
"""

from enum import Enum

import astropy.units as u
import numpy as np
from gammapy.catalog import CATALOG_REGISTRY
from gammapy.modeling import Parameter
from gammapy.modeling.models import (
    SPATIAL_MODEL_REGISTRY,
    SPECTRAL_MODEL_REGISTRY,
    DatasetModels,
    EBLAbsorptionNormSpectralModel,
    FoVBackgroundModel,
    Models,
    SkyModel,
    SpectralModel,
)

from asgardpy.base.base import AngleType, BaseConfig, FrameEnum, PathType
from asgardpy.base.geom import SkyPositionConfig

__all__ = [
    "BrokenPowerLaw2SpectralModel",
    "EBLAbsorptionModel",
    "ExpCutoffLogParabolaSpectralModel",
    "ModelTypeEnum",
    "RoISelectionConfig",
    "SpatialModelConfig",
    "SpectralModelConfig",
    "Target",
    "add_ebl_model_from_config",
    "apply_selection_mask_to_models",
    "config_to_dict",
    "get_models_from_catalog",
    "read_models_from_asgardpy_config",
    "set_models",
]


# Basic components to define the Target Config and any Models Config
class ModelTypeEnum(str, Enum):
    """
    Config section for Different Gammapy Model type.
    """

    skymodel = "SkyModel"
    fovbkgmodel = "FoVBackgroundModel"


class EBLAbsorptionModel(BaseConfig):
    """
    Config section for parameters to use for EBLAbsorptionNormSpectralModel.
    """

    filename: PathType = "None"
    reference: str = ""
    type: str = "EBLAbsorptionNormSpectralModel"
    redshift: float = 0.0
    alpha_norm: float = 1.0


class ModelParams(BaseConfig):
    """Config section for parameters to use for a basic Parameter object."""

    name: str = ""
    value: float = 1
    unit: str = " "
    error: float = 0.1
    min: float = 0.1
    max: float = 10
    frozen: bool = True


class SpectralModelConfig(BaseConfig):
    """
    Config section for parameters to use for creating a SpectralModel object.
    """

    type: str = ""
    parameters: list[ModelParams] = [ModelParams()]
    ebl_abs: EBLAbsorptionModel = EBLAbsorptionModel()


class SpatialModelConfig(BaseConfig):
    """
    Config section for parameters to use for creating a SpatialModel object.
    """

    type: str = ""
    frame: FrameEnum = FrameEnum.icrs
    parameters: list[ModelParams] = [ModelParams()]


class ModelComponent(BaseConfig):
    """Config section for parameters to use for creating a SkyModel object."""

    name: str = ""
    type: ModelTypeEnum = ModelTypeEnum.skymodel
    datasets_names: list[str] = [""]
    spectral: SpectralModelConfig = SpectralModelConfig()
    spatial: SpatialModelConfig = SpatialModelConfig()


class RoISelectionConfig(BaseConfig):
    """
    Config section for parameters to perform some selection on FoV source
    models.
    """

    roi_radius: AngleType = 0 * u.deg
    free_sources: list[str] = []


class CatalogConfig(BaseConfig):
    """Config section for parameters to use information from Catalog."""

    name: str = ""
    selection_radius: AngleType = 0 * u.deg
    exclusion_radius: AngleType = 0 * u.deg


class Target(BaseConfig):
    """Config section for main information on creating various Models."""

    source_name: str = ""
    sky_position: SkyPositionConfig = SkyPositionConfig()
    use_uniform_position: bool = True
    models_file: PathType = "None"
    datasets_with_fov_bkg_model: list[str] = []
    use_catalog: CatalogConfig = CatalogConfig()
    components: list[ModelComponent] = [ModelComponent()]
    covariance: str = ""
    from_3d: bool = False
    roi_selection: RoISelectionConfig = RoISelectionConfig()


class ExpCutoffLogParabolaSpectralModel(SpectralModel):
    r"""Spectral Exponential Cutoff Log Parabola model.

    Using a simple template from Gammapy.

    .. math::
        \phi(E) = \phi_0 \left( \frac{E}{E_0} \right) ^ {
          - \alpha_1 - \beta \log{ \left( \frac{E}{E_0} \right) }} \cdot
          \exp(- {(\lambda E})^{\alpha_2})

    Parameters
    ----------
    amplitude : `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : `~astropy.units.Quantity`
        :math:`E_0`
    alpha_1 : `~astropy.units.Quantity`
        :math:`\alpha_1`
    beta : `~astropy.units.Quantity`
        :math:`\beta`
    lambda_ : `~astropy.units.Quantity`
        :math:`\lambda`
    alpha_2 : `~astropy.units.Quantity`
        :math:`\alpha_2`
    """

    tag = ["ExpCutoffLogParabolaSpectralModel", "ECLP"]

    amplitude = Parameter(
        "amplitude",
        "1e-12 cm-2 s-1 TeV-1",
        scale_method="scale10",
        interp="log",
    )
    amplitude._is_norm = True
    reference = Parameter("reference", "1 TeV", frozen=True)
    alpha_1 = Parameter("alpha_1", -2)
    alpha_2 = Parameter("alpha_2", 1, frozen=True)
    beta = Parameter("beta", 1)
    lambda_ = Parameter("lambda_", "0.1 TeV-1")

    @staticmethod
    def evaluate(energy, amplitude, reference, alpha_1, beta, lambda_, alpha_2):
        """Evaluate the model (static function)."""
        en_ref = energy / reference
        exponent = -alpha_1 - beta * np.log(en_ref)
        cutoff = np.exp(-np.power(energy * lambda_, alpha_2))

        return amplitude * np.power(en_ref, exponent) * cutoff


class BrokenPowerLaw2SpectralModel(SpectralModel):
    r"""Spectral broken power-law 2 model.

    In this slightly modified Broken Power Law, instead of having the second index
    as a distinct parameter, the difference in the spectral indices around the
    Break Energy is used, to try for some assumptions on the different physical
    processes that define the full spectrum, where the second process is dependent
    on the first process.

    For more information see :ref:`broken-powerlaw-spectral-model`.

    .. math::
        \phi(E) = \phi_0 \cdot \begin{cases}
                \left( \frac{E}{E_{break}} \right)^{-\Gamma_1} & \text{if } E < E_{break} \\
                \left( \frac{E}{E_{break}} \right)^{-(\Gamma_1 + \Delta\Gamma)} & \text{otherwise}
            \end{cases}

    Parameters
    ----------
    index1 : `~astropy.units.Quantity`
        :math:`\Gamma_1`
    index_diff : `~astropy.units.Quantity`
        :math:`\Delta\Gamma`
    amplitude : `~astropy.units.Quantity`
        :math:`\phi_0`
    ebreak : `~astropy.units.Quantity`
        :math:`E_{break}`

    See Also
    --------
    SmoothBrokenPowerLawSpectralModel
    """

    tag = ["BrokenPowerLaw2SpectralModel", "bpl2"]
    index1 = Parameter("index1", 2.0)
    index_diff = Parameter("index_diff", 1.0)
    amplitude = Parameter(
        name="amplitude",
        value="1e-12 cm-2 s-1 TeV-1",
        scale_method="scale10",
        interp="log",
    )
    amplitude._is_norm = True
    ebreak = Parameter("ebreak", "1 TeV")

    @staticmethod
    def evaluate(energy, index1, index_diff, amplitude, ebreak):
        """Evaluate the model (static function)."""
        energy = np.atleast_1d(energy)
        cond = energy < ebreak
        bpwl2 = amplitude * np.ones(energy.shape)

        index2 = index1 + index_diff
        bpwl2[cond] *= (energy[cond] / ebreak) ** (-index1)
        bpwl2[~cond] *= (energy[~cond] / ebreak) ** (-index2)

        return bpwl2


SPECTRAL_MODEL_REGISTRY.append(ExpCutoffLogParabolaSpectralModel)
SPECTRAL_MODEL_REGISTRY.append(BrokenPowerLaw2SpectralModel)


# Function for Models assignment
def extend_bkg_models(models, all_datasets, datasets_with_fov_bkg_model):
    """
    Function for extending a Background Model for a given 3D dataset name.

    It checks if the given dataset is of MapDataset or MapDatasetOnOff type and
    no associated background model exists already.
    """
    if len(datasets_with_fov_bkg_model) > 0:
        bkg_models = []

        for dataset in all_datasets:
            if dataset.name in datasets_with_fov_bkg_model:
                if "MapDataset" in dataset.tag and dataset.background_model is None:
                    bkg_models.append(FoVBackgroundModel(dataset_name=dataset.name))

        models.extend(bkg_models)

    return models


def update_models_datasets_names(models, source_name, datasets_name_list):
    """
    Function to update the datasets_names list for the given list of models.

    If FoVBackgroundModel is provided, remove the -bkg part of the name of the
    dataset to get the appropriate datasets_name.
    """
    if len(models) > 1:
        models[source_name].datasets_names = datasets_name_list

        bkg_model_name = [m_name for m_name in models.names if "bkg" in m_name]

        if len(bkg_model_name) > 0:
            for bkg_name in bkg_model_name:
                models[bkg_name].datasets_names = [bkg_name[:-4]]
    else:
        models.datasets_names = datasets_name_list

    return models


def set_models(
    config_target,
    datasets,
    datasets_name_list=None,
    models=None,
):
    """
    Set models on given Datasets.

    Parameters
    ----------
    config_target: `AsgardpyConfig.target`
        AsgardpyConfig containing target information.
    datasets: `gammapy.datasets.Datasets`
        Datasets object
    dataset_name_list: list
        List of datasets_names to be used on the Models, before assigning them
        to the given datasets.
    models : `~gammapy.modeling.models.Models` or str of file location or None
        Models object or YAML models string

    Returns
    -------
    datasets: `gammapy.datasets.Datasets`
        Datasets object with Models assigned.
    """
    # Have some checks on argument types
    if isinstance(models, DatasetModels | list):
        models = Models(models)
    elif isinstance(models, str):
        models = Models.read(models)
    elif models is None:
        models = Models(models)
    else:
        raise TypeError(f"Invalid type: {type(models)}")

    if len(models) == 0:
        if config_target.components[0].name != "":
            models = read_models_from_asgardpy_config(config_target)
        else:
            raise ValueError("No input for Models provided for the Target source!")
    else:
        models = apply_selection_mask_to_models(
            list_sources=models,
            target_source=config_target.source_name,
            roi_radius=config_target.roi_selection.roi_radius,
            free_sources=config_target.roi_selection.free_sources,
        )

    models = extend_bkg_models(models, datasets, config_target.datasets_with_fov_bkg_model)

    if datasets_name_list is None:
        datasets_name_list = datasets.names

    models = update_models_datasets_names(models, config_target.source_name, datasets_name_list)

    datasets.models = models

    return datasets, models


def apply_models_mask_in_roi(list_sources_excluded, target_source, roi_radius, free_sources):
    """
    Function to apply a selection mask on a given list of models in a given RoI.

    The target source position is considered as the center of RoI.

    For a given list of non free sources, the spectral amplitude is left
    unfrozen or allowed to vary.
    """
    if not target_source:
        target_source = list_sources_excluded[0].name
        target_source_pos = list_sources_excluded[0].spatial_model.position
    else:
        target_source_pos = list_sources_excluded[target_source].spatial_model.position

    # If RoI radius is provided and is not default
    if roi_radius.to_value("deg") != 0:
        for model_ in list_sources_excluded:
            model_pos = model_.spatial_model.position
            separation = target_source_pos.separation(model_pos).to_value("deg")
            if separation >= roi_radius.to_value("deg"):
                model_.spectral_model.freeze()
    else:
        if len(free_sources) > 0:
            for model_ in list_sources_excluded:
                # Freeze all spectral parameters for other models
                if model_.name != target_source:
                    model_.spectral_model.freeze()
                # and now unfreeze the amplitude of selected models
                if model_.name in free_sources:
                    model_.spectral_model.parameters["amplitude"].frozen = False

    return list_sources_excluded


def apply_selection_mask_to_models(
    list_sources, target_source=None, selection_mask=None, roi_radius=0 * u.deg, free_sources=None
):
    """
    For a given list of source models, with a given target source, apply various
    selection masks on the Region of Interest in the sky. This will lead to
    complete exclusion of models or freezing some or all spectral parameters.
    These selections excludes the diffuse background models in the given list.

    First priority is given if a distinct selection mask is provided, with a
    list of excluded regions to return only the source models within the selected
    ROI.

    Second priority is on creating a Circular ROI as per the given radius, and
    freeze all the spectral parameters of the models of the sources.

    Third priority is when a list of free_sources is provided, then the
    spectral amplitude of models of those sources, if present in the given list
    of models, will be unfrozen, and hence, allowed to be variable for fitting.

    Parameters
    ----------
    list_sources: '~gammapy.modeling.models.Models'
        Models object containing a list of source models.
    target_source: 'str'
        Name of the target source, whose position is used as the center of ROI.
    selection_mask: 'WcsNDMap'
        Map containing a boolean mask to apply to Models object.
    roi_radius: 'astropy.units.Quantity' or 'asgardpy.data.base.AngleType'
        Radius for a circular region around ROI (deg)
    free_sources: 'list'
        List of source names for which the spectral amplitude is to be kept free.

    Returns
    -------
    list_sources: '~gammapy.modeling.models.Models'
        Selected Models object.
    """
    list_sources_excluded = []
    list_diffuse = []

    if free_sources is None:
        free_sources = []

    # Separate the list of sources and diffuse background
    for model_ in list_sources:
        if "diffuse" in model_.name or "bkg" in model_.name:
            list_diffuse.append(model_)
        else:
            list_sources_excluded.append(model_)

    list_sources_excluded = Models(list_sources_excluded)

    # If a distinct selection mask is provided
    if selection_mask:
        list_sources_excluded = list_sources_excluded.select_mask(selection_mask)

    list_sources_excluded = apply_models_mask_in_roi(
        list_sources_excluded, target_source, roi_radius, free_sources
    )

    # Add the diffuse background models back
    for diff_ in list_diffuse:
        list_sources_excluded.append(diff_)

    # Re-assign to the main variable
    list_sources = list_sources_excluded

    return list_sources


# Functions for Models generation
def add_ebl_model_from_config(spectral_model, model_config):
    """
    Function for adding the EBL model from a given AsgardpyConfig file to the
    given spectral model (assumed intrinsic).
    """
    ebl_model = model_config.spectral.ebl_abs

    # First check for filename of a custom EBL model
    if ebl_model.filename.is_file():
        model2 = EBLAbsorptionNormSpectralModel.read(str(ebl_model.filename), redshift=ebl_model.redshift)
        # Update the reference name when using the custom EBL model for logging
        ebl_model.reference = ebl_model.filename.name[:-8].replace("-", "_")
    else:
        model2 = EBLAbsorptionNormSpectralModel.read_builtin(ebl_model.reference, redshift=ebl_model.redshift)
    if ebl_model.alpha_norm:
        model2.alpha_norm.value = ebl_model.alpha_norm

    spectral_model *= model2

    return spectral_model


def read_models_from_asgardpy_config(config):
    """
    Reading Models information from AsgardpyConfig and return Models object.
    If a FoVBackgroundModel information is provided, it will also be added.

    Parameter
    ---------
    config: `AsgardpyConfig`
        Config section containing Target source information

    Returns
    -------
    models: `gammapy.modeling.models.Models`
        Full gammapy Models object.
    """
    models = Models()

    for model_config in config.components:
        # Spectral Model
        spectral_model = SPECTRAL_MODEL_REGISTRY.get_cls(model_config.spectral.type)().from_dict(
            {"spectral": config_to_dict(model_config.spectral)}
        )
        if model_config.spectral.ebl_abs.reference != "":
            spectral_model = add_ebl_model_from_config(spectral_model, model_config)

        spectral_model.name = config.source_name

        # Spatial model if provided
        if model_config.spatial.type != "":
            spatial_model = SPATIAL_MODEL_REGISTRY.get_cls(model_config.spatial.type)().from_dict(
                {"spatial": config_to_dict(model_config.spatial)}
            )
        else:
            spatial_model = None

        match model_config.type:
            case "SkyModel":
                models.append(
                    SkyModel(
                        spectral_model=spectral_model,
                        spatial_model=spatial_model,
                        name=config.source_name,
                    )
                )

            # FoVBackgroundModel is the second component
            case "FoVBackgroundModel":
                models.append(
                    FoVBackgroundModel(
                        dataset_name=model_config.datasets_names[0],
                        spectral_model=spectral_model,
                        spatial_model=spatial_model,
                    )
                )

    return models


def config_to_dict(model_config):
    """
    Convert the Spectral/Spatial models into dict.
    Probably an extra step and maybe removed later.

    Parameter
    ---------
    model_config: `AsgardpyConfig`
        Config section containing Target Model SkyModel components only.

    Returns
    -------
    model_dict: dict
        dictionary of the particular model.
    """
    model_dict = {}
    model_dict["type"] = str(model_config.type)
    model_dict["parameters"] = []

    for par in model_config.parameters:
        par_dict = {}
        par_dict["name"] = par.name
        par_dict["value"] = par.value
        par_dict["unit"] = par.unit
        par_dict["error"] = par.error
        par_dict["min"] = par.min
        par_dict["max"] = par.max
        par_dict["frozen"] = par.frozen
        model_dict["parameters"].append(par_dict)

    # For spatial model, include frame info
    if hasattr(model_config, "frame"):
        model_dict["frame"] = model_config.frame

    return model_dict


def get_models_from_catalog(config_target, center_position_gal):
    """
    From a given catalog in Gammapy, get a list of source SkyModels around the
    target source as per the config information.

    Parameters
    ----------
    config_target: `asgardpy.config.generator.AsgardpyConfig.target`
        Config information on the target source.
    center_position_gal: `astropy.coordinates.Galactic`
        Central location of the target source in galactic coordinates.

    Returns
    list_source_models: `list`
        List of catalog source models around the target source.
    """
    list_source_models = []

    # Read the SkyModel info from AsgardpyConfig.target section
    if len(config_target.components) > 0:
        models_ = read_models_from_asgardpy_config(config_target)
        list_source_models = models_

    # Check if a catalog data is given with selection radius
    if config_target.use_catalog.selection_radius != 0 * u.deg:
        catalog = CATALOG_REGISTRY.get_cls(config_target.use_catalog.name)()

        sep = catalog.positions.separation(center_position_gal)

        for k, cat_ in enumerate(catalog):
            if sep[k] < config_target.use_catalog.selection_radius:
                list_source_models.append(cat_.sky_model())

    return list_source_models
