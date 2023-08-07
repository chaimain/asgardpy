"""
Classes containing the Target config parameters for the high-level interface and
also the functions involving Models generation and assignment to datasets.
"""

from typing import List

import astropy.units as u
import numpy as np
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
    create_fermi_isotropic_diffuse_model,
)

from asgardpy.base.base import AngleType, BaseConfig, FrameEnum, PathType
from asgardpy.base.geom import SkyPositionConfig
from asgardpy.gammapy.interoperate_models import (
    get_gammapy_spectral_model,
    xml_spatial_model_to_gammapy,
    xml_spectral_model_to_gammapy,
)

__all__ = [
    "BrokenPowerLaw2SpectralModel",
    "EBLAbsorptionModel",
    "ExpCutoffLogParabolaSpectralModel",
    "RoISelectionConfig",
    "SpatialModelConfig",
    "SpectralModelConfig",
    "Target",
    "apply_selection_mask_to_models",
    "config_to_dict",
    "create_gal_diffuse_skymodel",
    "create_iso_diffuse_skymodel",
    "create_source_skymodel",
    "read_models_from_asgardpy_config",
    "set_models",
]


# Basic components to define the Target Config and any Models Config
class EBLAbsorptionModel(BaseConfig):
    """
    Config section for parameters to use for EBLAbsorptionNormSpectralModel.
    """

    filename: PathType = "."
    reference: str = "dominguez"
    type: str = "EBLAbsorptionNormSpectralModel"
    redshift: float = 0.4
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
    parameters: List[ModelParams] = [ModelParams()]
    ebl_abs: EBLAbsorptionModel = EBLAbsorptionModel()


class SpatialModelConfig(BaseConfig):
    """
    Config section for parameters to use for creating a SpatialModel object.
    """

    type: str = ""
    frame: FrameEnum = FrameEnum.icrs
    parameters: List[ModelParams] = [ModelParams()]


class SkyModelComponent(BaseConfig):
    """Config section for parameters to use for creating a SkyModel object."""

    name: str = ""
    type: str = "SkyModel"
    spectral: SpectralModelConfig = SpectralModelConfig()
    spatial: SpatialModelConfig = SpatialModelConfig()


class RoISelectionConfig(BaseConfig):
    """
    Config section for parameters to perform some selection on FoV source
    models.
    """

    roi_radius: AngleType = 0 * u.deg
    free_sources: List[str] = []


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
    models_file: PathType = "."
    add_fov_bkg_model: bool = False
    use_catalog: CatalogConfig = CatalogConfig()
    components: List[SkyModelComponent] = [SkyModelComponent()]
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
        is_norm=True,
    )
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
        is_norm=True,
    )
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
    dataset_name_list: List
        List of datasets_names to be used on the Models, before assigning them
        to the given datasets.
    models : `~gammapy.modeling.models.Models` or str
        Models object or YAML models string

    Returns
    -------
    datasets: `gammapy.datasets.Datasets`
        Datasets object with Models assigned.
    """
    # Have some checks on argument types
    if isinstance(models, (DatasetModels, list)):
        models = Models(models)
    elif isinstance(models, PathType()):
        models = Models.read(models)
    else:
        raise TypeError(f"Invalid type: {type(models)}")

    if len(models) == 0:
        if len(config_target.components) > 0:
            models = read_models_from_asgardpy_config(config_target)
        else:
            raise Exception("No input for Models provided for the Target source!")
    else:
        models = apply_selection_mask_to_models(
            list_sources=models,
            target_source=config_target.source_name,
            roi_radius=config_target.roi_selection.roi_radius,
            free_sources=config_target.roi_selection.free_sources,
        )

    if config_target.add_fov_bkg_model:
        # For extending a Background Model for each 3D dataset
        bkg_models = []

        for dataset in datasets:
            if dataset.tag == "MapDataset" and dataset.background_model is None:
                bkg_models.append(FoVBackgroundModel(dataset_name=f"{dataset.name}-bkg"))

        models.extend(bkg_models)

    if datasets_name_list is None:
        datasets_name_list = datasets.names

    if len(models) > 1:
        models[config_target.source_name].datasets_names = datasets_name_list

        # Check if FoVBackgroundModel is provided
        bkg_model_name = [m_name for m_name in models.names if "bkg" in m_name]
        if len(bkg_model_name) > 0:
            # Remove the -bkg part of the name of the FoVBackgroundModel to get
            # the appropriate datasets name
            models[bkg_model_name].datasets_names = [bkg_model_name[:-4]]
    else:
        models.datasets_names = datasets_name_list

    datasets.models = models

    return datasets, models


def apply_selection_mask_to_models(
    list_sources, target_source=None, selection_mask=None, roi_radius=0 * u.deg, free_sources=[]
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

    # Separate the list of sources and diffuse background
    for model_ in list_sources:
        if "diffuse" in model_.name or "bkg" in model_.name:
            list_diffuse.append(model_)
        else:
            list_sources_excluded.append(model_)

    list_sources_excluded = Models(list_sources_excluded)

    # Get the target source position as the center of RoI
    if not target_source:
        target_source = list_sources_excluded[0].name
        target_source_pos = target_source.spatial_model.position
    else:
        target_source_pos = list_sources_excluded[target_source].spatial_model.position

    # If a distinct selection mask is provided
    if selection_mask:
        list_sources_excluded = list_sources_excluded.select_mask(selection_mask)

    # If RoI radius is provided and is not default
    if roi_radius.to_value("deg") != 0:
        for model_ in list_sources_excluded:
            model_pos = model_.spatial_model.position
            separation = target_source_pos.separation(model_pos).to_value("deg")
            if separation >= roi_radius.to_value("deg"):
                model_.spectral_model.freeze()
    else:
        # For a given list of non free sources, unfreeze the spectral amplitude
        if len(free_sources) > 0:
            for model_ in list_sources_excluded:
                # Freeze all spectral parameters for other models
                if model_.name != target_source:
                    model_.spectral_model.freeze()
                # and now unfreeze the amplitude of selected models
                if model_.name in free_sources:
                    model_.spectral_model.parameters["amplitude"].frozen = False

    # Add the diffuse background models back
    for diff_ in list_diffuse:
        list_sources_excluded.append(diff_)

    # Re-assign to the main variable
    list_sources = list_sources_excluded

    return list_sources


# Functions for Models generation
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
    # SkyModel is the first component
    model_config = config.components[0]

    # Spectral Model
    if model_config.spectral.ebl_abs.reference != "":
        model1 = SPECTRAL_MODEL_REGISTRY.get_cls(model_config.spectral.type)().from_dict(
            {"spectral": config_to_dict(model_config.spectral)}
        )

        ebl_model = model_config.spectral.ebl_abs

        # First check for filename of a custom EBL model
        if ebl_model.filename.is_file():
            model2 = EBLAbsorptionNormSpectralModel.read(
                str(ebl_model.filename), redshift=ebl_model.redshift
            )
            # Update the reference name when using the custom EBL model for logging
            ebl_model.reference = ebl_model.filename.name[:-8].replace("-", "_")
        else:
            model2 = EBLAbsorptionNormSpectralModel.read_builtin(
                ebl_model.reference, redshift=ebl_model.redshift
            )
        if ebl_model.alpha_norm:
            model2.alpha_norm.value = ebl_model.alpha_norm
        spectral_model = model1 * model2
    else:
        if model_config.spectral.type == "ExpCutoffLogParabolaSpectralModel":
            spectral_model = ExpCutoffLogParabolaSpectralModel().from_dict(
                {"spectral": config_to_dict(model_config.spectral)}
            )
        elif model_config.spectral.type == "BrokenPowerLaw2SpectralModel":
            model1 = BrokenPowerLaw2SpectralModel().from_dict(
                {"spectral": config_to_dict(model_config.spectral)}
            )
        else:
            spectral_model = SPECTRAL_MODEL_REGISTRY.get_cls(
                model_config.spectral.type
            )().from_dict({"spectral": config_to_dict(model_config.spectral)})
    spectral_model.name = config.source_name

    # Spatial model if provided
    if model_config.spatial.type != "":
        spatial_model = SPATIAL_MODEL_REGISTRY.get_cls(model_config.spatial.type)().from_dict(
            {"spatial": config_to_dict(model_config.spatial)}
        )
    else:
        spatial_model = None

    models = Models(
        [
            SkyModel(
                spectral_model=spectral_model,
                spatial_model=spatial_model,
                name=config.source_name,
            )
        ]
    )

    if len(config.components) > 1:
        # FoVBackgroundModel is the second component
        model_config_fov = config.components[1]

        # Spectral Model. At least this has to be provided for distinct
        # parameter values
        spectral_model_fov = SPECTRAL_MODEL_REGISTRY.get_cls(
            model_config_fov.spectral.type
        )().from_dict({"spectral": config_to_dict(model_config_fov.spectral)})

        # Spatial model if provided
        if model_config_fov.spatial.type != "":
            spatial_model_fov = SPATIAL_MODEL_REGISTRY.get_cls(
                model_config_fov.spatial.type
            )().from_dict({"spatial": config_to_dict(model_config_fov.spatial)})
        else:
            spatial_model_fov = None

        model_fov = FoVBackgroundModel(
            spectral_model=spectral_model_fov,
            spatial_model=spatial_model_fov,
            name=model_config_fov.name,
        )
        models.append(model_fov)

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
    try:
        frame_ = getattr(model_dict, "frame")
        model_dict["frame"] = frame_
    except AttributeError:
        pass

    return model_dict


def create_source_skymodel(config_target, source, aux_path, base_model_type="Fermi-XML"):
    """
    Build SkyModels from a given AsgardpyConfig section of the target
    source information, list of XML file and other relevant information.

    Parameters
    ----------
    config_target: `AsgardpyConfig`
        Config section containing the Target source information.
    source: dict
        Dictionary containing the source models information from XML file.
    aux_path: str
        Path location of the DL3 auxiliary files.
    base_model_type: str
        Name indicating the XML format used to read the skymodels from.

    Returns
    -------
    source_sky_model: `gammapy.modeling.SkyModel`
        SkyModels object for the given source information.
    is_source_target: bool
        Boolean to check if the Models belong to the target source.
    """
    # Following a general XML format, taking Fermi-LAT as reference.
    source_name = source["@name"]
    spectrum_type = source["spectrum"]["@type"]
    spectrum = source["spectrum"]["parameter"]

    source_name_check = source_name.replace("_", "").replace(" ", "")
    target_check = config_target.source_name.replace("_", "").replace(" ", "")

    # initialized to check for the case if target spectral model information
    # is to be taken from the Config
    spectral_model = None

    # Check if target_source file exists
    is_source_target = False
    ebl_atten = False

    # If Target source model's spectral component is to be taken from Config
    # and not from 3D dataset.
    if source_name_check == target_check:
        source_name = config_target.source_name
        is_source_target = True

        # Only taking the spectral model information right now.
        if not config_target.from_3d:
            models_ = read_models_from_asgardpy_config(config_target)
            spectral_model = models_[0].spectral_model

    if spectral_model is None:
        # Define the Spectral Model type for Gammapy
        spectral_model, ebl_atten = get_gammapy_spectral_model(
            spectrum_type,
            ebl_atten,
            base_model_type,
        )
        spectrum_type = spectrum_type.split("EblAtten::")[-1]

        # Read the parameter values from XML file to create SpectralModel
        params_list = xml_spectral_model_to_gammapy(
            spectrum,
            spectrum_type,
            is_target=is_source_target,
            keep_sign=ebl_atten,
            base_model_type=base_model_type,
        )

        for param_ in params_list:
            setattr(spectral_model, param_.name, param_)
        config_spectral = config_target.components[0].spectral
        ebl_absorption_included = config_spectral.ebl_abs is not None

        if is_source_target and ebl_absorption_included:
            ebl_model = config_spectral.ebl_abs

            if ebl_model.filename.is_file():
                ebl_spectral_model = EBLAbsorptionNormSpectralModel.read(
                    str(ebl_model.filename), redshift=ebl_model.redshift
                )
                ebl_model.reference = ebl_model.filename.name[:-8].replace("-", "_")
            else:
                ebl_spectral_model = EBLAbsorptionNormSpectralModel.read_builtin(
                    ebl_model.reference, redshift=ebl_model.redshift
                )
            spectral_model = spectral_model * ebl_spectral_model

    # Reading Spatial model from the XML file
    spatial_model = xml_spatial_model_to_gammapy(aux_path, source["spatialModel"], base_model_type)

    spatial_model.freeze()
    source_sky_model = SkyModel(
        spectral_model=spectral_model,
        spatial_model=spatial_model,
        name=source_name,
    )

    return source_sky_model, is_source_target


def create_iso_diffuse_skymodel(iso_file, key):
    """
    Create a SkyModel of the Fermi Isotropic Diffuse Model and assigning
    name as per the observation key. If there are no distinct key types of
    files, the value is None.
    """
    diff_iso = create_fermi_isotropic_diffuse_model(
        filename=iso_file, interp_kwargs={"fill_value": None}
    )
    if key:
        diff_iso._name = f"{diff_iso.name}-{key}"

    # Parameters' limits
    diff_iso.spectral_model.model1.parameters[0].min = 0.001
    diff_iso.spectral_model.model1.parameters[0].max = 10
    diff_iso.spectral_model.model2.parameters[0].min = 0
    diff_iso.spectral_model.model2.parameters[0].max = 10

    return diff_iso


def create_gal_diffuse_skymodel(diff_gal):
    """
    Create SkyModel of the Diffuse Galactic sources.
    """
    template_diffuse = SPATIAL_MODEL_REGISTRY.get_cls("TemplateSpatialModel")(
        diff_gal, normalize=False, filename=diff_gal.meta["filename"]
    )
    source = SkyModel(
        spectral_model=SPECTRAL_MODEL_REGISTRY.get_cls("PowerLawNormSpectralModel")(),
        spatial_model=template_diffuse,
        name="diffuse-iem",
    )
    source.parameters["norm"].min = 0
    source.parameters["norm"].max = 10
    source.parameters["norm"].frozen = False

    return source
