"""
Classes containing the Target config parameters for the high-level interface and
also the functions involving Models generation and assignment to datasets.
"""

from typing import List

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from gammapy.maps import Map
from gammapy.modeling import Parameter, Parameters
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
from scipy.stats import chi2, norm

from asgardpy.base import AngleType, BaseConfig, PathType, SkyPositionConfig

__all__ = [
    "BrokenPowerLaw2SpectralModel",
    "EBLAbsorptionModel",
    "ExpCutoffLogParabolaSpectralModel",
    "RoISelectionConfig",
    "SpatialModelConfig",
    "SpectralModelConfig",
    "Target",
    "apply_selection_mask_to_models",
    "check_model_preference_aic",
    "check_model_preference_lrt",
    "config_to_dict",
    "create_gal_diffuse_skymodel",
    "create_iso_diffuse_skymodel",
    "create_source_skymodel",
    "get_chi2_pval",
    "params_renaming_to_gammapy",
    "params_rescale_to_gammapy",
    "read_models_from_asgardpy_config",
    "set_models",
    "xml_spatial_model_to_gammapy",
    "xml_spectral_model_to_gammapy_params",
]


# Basic components to define the Target Config and any Models Config
class EBLAbsorptionModel(BaseConfig):
    """
    Config section for parameters to use for EBLAbsorptionNormSpectralModel.
    """

    filename: PathType = PathType(".")
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
    models_file: PathType = PathType(".")
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
    elif isinstance(models, PathType):
        models = Models.read(models)
    else:
        raise TypeError(f"Invalid type: {type(models)}")

    if len(models) == 0:
        if len(config_target.components) > 0:
            spectral_model, spatial_model = read_models_from_asgardpy_config(config_target)
            models = Models(
                SkyModel(
                    spectral_model=spectral_model,
                    spatial_model=spatial_model,
                    name=config_target.source_name,
                )
            )
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
                bkg_models.append(FoVBackgroundModel(dataset_name=dataset.name))

        models.extend(bkg_models)

    if datasets_name_list is None:
        datasets_name_list = datasets.names

    if len(models) > 1:
        models[config_target.source_name].datasets_names = datasets_name_list
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
        if "diffuse" in model_.name:
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
            separation = target_source_pos.separation(model_pos).deg
            if separation >= roi_radius.deg:
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
    Reading Models information from AsgardpyConfig and return Spectral and
    Spatial Models object to be combined later into SkyModels/Models object.

    Parameter
    ---------
    config: `AsgardpyConfig`
        Config section containing Target source information

    Returns
    -------
    spectral_model: `gammapy.modeling.models.SpectralModel`
        Spectral Model components of a gammapy SkyModel object.
    spatial_model: `gammapy.modeling.models.SpatialModel`
        Spatial Model components of a gammapy SkyModel object.
    """
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

    return spectral_model, spatial_model


def config_to_dict(model_config):
    """
    Convert the Spectral/Spatial models into dict.
    Probably an extra step and maybe removed later.

    Parameter
    ---------
    model_config: `AsgardpyConfig`
        Config section containg Target Model SkyModel components only.

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

    return model_dict


def params_renaming_to_gammapy(
    params_1_name, params_gpy, spectrum_type, params_1_base_model="Fermi-XML"
):
    """
    Reading from a given parameter name, get basic parameters details like name,
    unit and is_norm as per Gammapy definition.

    This function collects all such switch cases, based on the base model of the
    given set of parameters.
    """
    if params_1_base_model == "Fermi-XML":
        if params_1_name in ["norm", "prefactor", "integral"]:
            params_gpy["name"] = "amplitude"
            params_gpy["unit"] = "cm-2 s-1 TeV-1"
            params_gpy["is_norm"] = True

        if params_1_name in ["scale", "eb"]:
            params_gpy["name"] = "reference"

        if params_1_name in ["breakvalue"]:
            params_gpy["name"] = "ebreak"

        if params_1_name in ["lowerlimit"]:
            params_gpy["name"] = "emin"

        if params_1_name in ["upperlimit"]:
            params_gpy["name"] = "emax"

        if params_1_name in ["cutoff", "expfactor"]:
            params_gpy["name"] = "lambda_"
            params_gpy["unit"] = "TeV-1"

        if params_1_name in ["index"]:
            params_gpy["name"] = "index"

        if params_1_name in ["index1"]:
            if spectrum_type in ["PLSuperExpCutoff", "PLSuperExpCutoff2"]:
                params_gpy["name"] = "index"
            else:
                params_gpy["name"] = "index1"  # For spectrum_type == "BrokenPowerLaw"

        if params_1_name in ["indexs"]:
            params_gpy["name"] = "index_1"  # For spectrum_type == "PLSuperExpCutoff4"

        if params_1_name in ["index2"]:
            if spectrum_type == "PLSuperExpCutoff4":
                params_gpy["name"] = "index_2"
            elif spectrum_type in ["PLSuperExpCutoff", "PLSuperExpCutoff2"]:
                params_gpy["name"] = "alpha"
            else:
                params_gpy["name"] = "index2"  # For spectrum_type == "BrokenPowerLaw"

        if params_1_name in ["expfactors"]:
            params_gpy["name"] = "expfactor"

    return params_gpy


def params_rescale_to_gammapy(params_gpy, spectrum_type, en_scale_1_to_gpy=1.0e-6, keep_sign=False):
    """
    Rescaling parameters to be used with Gammapy standard units, by using the
    various physical factors (energy for now), compared with the initial set of
    parameters as compared with Gammapy standard units.

    Also, scales the value, min and max of the given parameters, depending on
    their Parameter names.
    """
    if params_gpy["name"] in ["reference", "ebreak", "emin", "emax"]:
        params_gpy["unit"] = "TeV"
        params_gpy["value"] = (
            float(params_gpy["value"]) * float(params_gpy["scale"]) * en_scale_1_to_gpy
        )
        if "error" in params_gpy:
            params_gpy["error"] = (
                float(params_gpy["error"]) * float(params_gpy["scale"]) * en_scale_1_to_gpy
            )
        params_gpy["min"] = (
            float(params_gpy["min"]) * float(params_gpy["scale"]) * en_scale_1_to_gpy
        )
        params_gpy["max"] = (
            float(params_gpy["max"]) * float(params_gpy["scale"]) * en_scale_1_to_gpy
        )
        params_gpy["scale"] = 1.0

    if params_gpy["name"] in ["amplitude"]:
        params_gpy["value"] = (
            float(params_gpy["value"]) * float(params_gpy["scale"]) / en_scale_1_to_gpy
        )
        if "error" in params_gpy:
            params_gpy["error"] = (
                float(params_gpy["error"]) * float(params_gpy["scale"]) / en_scale_1_to_gpy
            )
        params_gpy["min"] = (
            float(params_gpy["min"]) * float(params_gpy["scale"]) / en_scale_1_to_gpy
        )
        params_gpy["max"] = (
            float(params_gpy["max"]) * float(params_gpy["scale"]) / en_scale_1_to_gpy
        )
        params_gpy["scale"] = 1.0

    if params_gpy["name"] in ["index", "index_1", "index_2", "beta"] and not keep_sign:
        # Other than EBL Attenuated Power Law?
        # spectral indices in gammapy are always taken as positive values.
        val_ = float(params_gpy["value"]) * float(params_gpy["scale"])
        if val_ < 0:
            params_gpy["value"] = -1 * val_

            # Reverse the limits while changing the sign
            min_ = -1 * float(params_gpy["min"]) * float(params_gpy["scale"])
            max_ = -1 * float(params_gpy["max"]) * float(params_gpy["scale"])
            params_gpy["min"] = min(min_, max_)
            params_gpy["max"] = max(min_, max_)
            params_gpy["scale"] = 1.0

    if params_gpy["name"] in ["lambda_"] and spectrum_type == "PLSuperExpCutoff":
        # Original parameter is inverse of what gammapy uses
        val_ = float(params_gpy["value"]) * float(params_gpy["scale"])
        params_gpy["value"] = en_scale_1_to_gpy / val_
        if "error" in params_gpy:
            params_gpy["error"] = en_scale_1_to_gpy * float(params_gpy["error"]) / (val_**2)
        min_ = en_scale_1_to_gpy / (float(params_gpy["min"]) * float(params_gpy["scale"]))
        max_ = en_scale_1_to_gpy / (float(params_gpy["max"]) * float(params_gpy["scale"]))
        params_gpy["min"] = min(min_, max_)
        params_gpy["max"] = max(min_, max_)
        params_gpy["scale"] = 1.0

    if float(params_gpy["scale"]) != 1.0:
        # Without any other modifications, but using the scale value
        params_gpy["value"] = float(params_gpy["value"]) * float(params_gpy["scale"])
        if "error" in params_gpy:
            params_gpy["error"] = float(params_gpy["error"]) * float(params_gpy["scale"])
        params_gpy["min"] = float(params_gpy["min"]) * float(params_gpy["scale"])
        params_gpy["max"] = float(params_gpy["max"]) * float(params_gpy["scale"])
        params_gpy["scale"] = 1.0

    return params_gpy


def xml_spectral_model_to_gammapy_params(params, spectrum_type, is_target=False, keep_sign=False):
    """
    Convert the Spectral Models information from XML model of FermiTools to Gammapy
    standards and return Parameters list.
    Details of the XML model can be seen at
    https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html
    and with examples at
    https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/xml_model_defs.html

    Models from the XML model, not read are -
    ExpCutoff, BPLExpCutoff, PLSuperExpCutoff3, Gaussian, BandFunction

    Parameters
    ----------
    params: `gammapy.modeling.Parameters`
        List of gammapy Parameter object of a particular Model
    spectrum_type: str
        Spectrum type as defined in XML. To be used only for special cases like
        PLSuperExpCutoff, PLSuperExpCutoff2 and PLSuperExpCutoff4
    is_target: bool
        Boolean to check if the given list of Parameters belong to the target
        source model or not.
    keep_sign: bool
        Boolean to keep the same sign on the parameter values or not.

    Returns
    -------
    params_final: `gammapy.modeling.Parameters`
        Final list of gammapy Parameter object
    """
    new_params = []

    for par in params:
        new_par = {}

        for key_ in par.keys():
            # Getting the "@par_name" for each parameter without the "@"
            if key_ != "@free":
                new_par[key_[1:].lower()] = par[key_]
            else:
                if par["@name"].lower() not in ["scale", "eb"]:
                    new_par["frozen"] = (par[key_] == "0") and not is_target
                else:
                    # Never change frozen status of Reference Energy
                    new_par["frozen"] = par[key_] == "0"
            new_par["unit"] = ""
            new_par["is_norm"] = False

            # Using the nomenclature as used in Gammapy
            new_par = params_renaming_to_gammapy(
                par["@name"].lower(), new_par, spectrum_type, params_1_base_model="Fermi-XML"
            )

        # Some modifications on scaling/sign:
        new_par = params_rescale_to_gammapy(
            new_par, spectrum_type, en_scale_1_to_gpy=1.0e-6, keep_sign=keep_sign
        )

        if new_par["name"] == "alpha" and spectrum_type in [
            "PLSuperExpCutoff",
            "PLSuperExpCutoff2",
        ]:
            new_par["frozen"] = par["@free"] == "0"

        # Read into Gammapy Parameter object
        new_param = Parameter(name=new_par["name"], value=new_par["value"])
        if "error" in new_par:
            new_param.error = new_par["error"]
        new_param.min = new_par["min"]
        new_param.max = new_par["max"]
        new_param.unit = new_par["unit"]
        new_param.frozen = new_par["frozen"]
        new_param._is_norm = new_par["is_norm"]

        new_params.append(new_param)

    params_final2 = Parameters(new_params)

    # Modifications when different parameters are interconnected
    if spectrum_type == "PLSuperExpCutoff2":
        alpha_inv = 1 / params_final2["alpha"].value
        min_sign = 1
        if params_final2["lambda_"].min < 0:
            min_sign = -1

        params_final2["lambda_"].value = params_final2["lambda_"].value ** alpha_inv * 1.0e6
        params_final2["lambda_"].min = min_sign * (
            abs(params_final2["lambda_"].min) ** alpha_inv * 1.0e6
        )
        params_final2["lambda_"].max = params_final2["lambda_"].max ** alpha_inv * 1.0e6

    return params_final2


def xml_spatial_model_to_gammapy(aux_path, xml_spatial_model):
    """
    Read the spatial model component of the XMl model to Gammapy SpatialModel
    object.

    Details of the XML model can be seen at
    https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html
    and with examples at
    https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/xml_model_defs.html

    Paramaters
    ----------
    aux_path: `Path`
        Path to the template diffuse models
    xml_spatial_model: `dict`
        Spatial Model component of a particular source from the XML file

    Returns
    -------
    spatial_model: `gammapy.modeling.models.SpatialModel`
        Gammapy Spatial Model object
    """
    spatial_pars = xml_spatial_model["parameter"]

    if xml_spatial_model["@type"] == "SkyDirFunction":
        for par_ in spatial_pars:
            if par_["@name"] == "RA":
                lon_0 = f"{par_['@value']} deg"
            if par_["@name"] == "DEC":
                lat_0 = f"{par_['@value']} deg"
        fk5_frame = SkyCoord(
            lon_0,
            lat_0,
            frame="fk5",
        )
        gal_frame = fk5_frame.transform_to("galactic")
        spatial_model = SPATIAL_MODEL_REGISTRY.get_cls("PointSpatialModel")().from_position(
            gal_frame
        )

    elif xml_spatial_model["@type"] == "SpatialMap":
        file_name = xml_spatial_model["@file"].split("/")[-1]
        file_path = aux_path / f"Templates/{file_name}"

        spatial_map = Map.read(file_path)
        spatial_map = spatial_map.copy(unit="sr^-1")

        spatial_model = SPATIAL_MODEL_REGISTRY.get_cls("TemplateSpatialModel")(
            spatial_map, filename=file_path
        )

    elif xml_spatial_model["@type"] == "RadialGaussian":
        for par_ in spatial_pars:
            if par_["@name"] == "RA":
                lon_0 = f"{par_['@value']} deg"
            if par_["@name"] == "DEC":
                lat_0 = f"{par_['@value']} deg"
            if par_["@name"] == "Sigma":
                sigma = f"{par_['@value']} deg"

        fk5_frame = SkyCoord(
            lon_0,
            lat_0,
            frame="fk5",
        )
        gal_frame = fk5_frame.transform_to("galactic")

        spatial_model = SPATIAL_MODEL_REGISTRY.get_cls("GaussianSpatialModel")(
            lon_0=gal_frame.l, lat_0=gal_frame.b, sigma=sigma, frame="galactic"
        )

    return spatial_model


def create_source_skymodel(config_target, source, aux_path):
    """
    Build SkyModels from a given AsgardpyConfig section of the target
    source information, list of LAT files and other relevant information.

    Parameters
    ----------
    config_target: `AsgardpyConfig`
        Config section containing the Target source information.
    source: dict
        Dictionary containing the source models infromation from XML file.
    aux_path: str
        Path location of the LAT auxillary files.

    Returns
    -------
    source_sky_model: `gammapy.modeling.SkyModel`
        SkyModels object for the given source information.
    is_source_target: bool
        Boolean to check if the Models belong to the target source.
    """
    source_name = source["@name"]
    spectrum_type = source["spectrum"]["@type"].split("EblAtten::")[-1]
    spectrum = source["spectrum"]["parameter"]

    source_name_check = source_name.replace("_", "").replace(" ", "")
    target_check = config_target.source_name.replace("_", "").replace(" ", "")

    # initialized to check for the case if target spectral model information
    # is to be taken from the Config
    spectral_model = None

    # Check if target_source file exists
    is_source_target = False
    ebl_atten_pl = False

    # If Target source model's spectral component is to be taken from Config
    # and not from 3D dataset.
    if source_name_check == target_check:
        source_name = config_target.source_name
        is_source_target = True

        # Only taking the spectral model information right now.
        if not config_target.from_3d:
            spectral_model, _ = read_models_from_asgardpy_config(config_target)

    if spectral_model is None:
        # Define the Spectral Model type for Gammapy
        for spec in spectrum:
            if spec["@name"] not in ["GalDiffModel", "IsoDiffModel"]:
                if spectrum_type in ["PLSuperExpCutoff", "PLSuperExpCutoff2"]:
                    spectrum_type_final = "ExpCutoffPowerLawSpectralModel"
                elif spectrum_type == "PLSuperExpCutoff4":
                    spectrum_type_final = "SuperExpCutoffPowerLaw4FGLDR3SpectralModel"
                else:
                    spectrum_type_final = f"{spectrum_type}SpectralModel"

                spectral_model = SPECTRAL_MODEL_REGISTRY.get_cls(spectrum_type_final)()

                if spectrum_type == "LogParabola":
                    if "EblAtten" in source["spectrum"]["@type"]:
                        spectral_model = SPECTRAL_MODEL_REGISTRY.get_cls("PowerLawSpectralModel")()
                        ebl_atten_pl = True
                    else:
                        spectral_model = SPECTRAL_MODEL_REGISTRY.get_cls(
                            "LogParabolaSpectralModel"
                        )()

        # Read the parameter values from XML file to create SpectralModel
        params_list = xml_spectral_model_to_gammapy_params(
            spectrum,
            spectrum_type,
            is_target=is_source_target,
            keep_sign=ebl_atten_pl,
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
    spatial_model = xml_spatial_model_to_gammapy(aux_path, source["spatialModel"])

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


def get_chi2_pval(stat_h0, stat_h1, ndof):
    """
    Using the log-likelihood value for the null hypothesis (H0) and the
    alternate hypothesis (H1), along with the total degrees of freedom,
    evaluate the chi2 value along with the p-value for the fitting statistics.

    In Gammapy, for 3D analysis, cash statistics is used, while for 1D analysis,
    wstat statistics is used. Check the documentation for more details
    https://docs.gammapy.org/1.1/user-guide/stats/index.html

    The null hypothesis here is the total statistics of the DL4 dataset without
    using any model information, and just using the background (3D) or OFF
    counts (1D) information to evaluate the likelihood value for the signal.

    The alternate hypothesis is by adding the Model parameters, to the
    likelihood evaluation of the signal. 

    Parameters
    ----------
    stat_h0: float
        log-likelihood value of the null hypothesis.
    stat_h1: float
        log-likelihood value of the alternate hypothesis.
    ndof: int
        Total number of degrees of freedom.

    Returns
    -------
    chi2_: float
        significance (Chi2) of the likelihood of primary fit model estimated in
        Gaussian distribution.
    pval: float
        p-value or the surviving probability...

    """
    pval = chi2.sf(stat_h0 - stat_h1, ndof)
    chi2_ = norm.isf(pval / 2)

    if not np.isfinite(chi2_):
        chi2_ = np.sqrt((stat_h0 - stat_h1))

    return chi2_, pval


def check_model_preference_lrt(stat_1, stat_2, ndof_1, ndof_2):
    """
    Log-likelihood ratio test. Checking the preference of a "nested" spectral
    model2 (observed), over a primary model1.

    Parameters
    ----------
    stat_1: `gammapy.modeling.fit.FitResult.total_stat`
        The total stat of the Fit result of the primary spectral model.
    stat_2: `gammapy.modeling.fit.FitResult.total_stat`
        The total stat of the Fit result of the nested spectral model.
    ndof_1: 'int'
        Number of energy bins used for spectral fit - number of free spectral parameters for the primary model
    ndof_2: 'int'
        Number of energy bins used for spectral fit - number of free spectral parameters for the nested model

    Returns
    -------
    p_value: float
        p-value for the ratio of the likelihoods
    gaussian_sigmas: float
        significance (Chi2) of the ratio of the likelihoods estimated in
        Gaussian distribution.
    n_dof: int
        number of degrees of freedom or free parameters between primary and
        nested model.
    """
    n_dof = ndof_2 - ndof_1

    if n_dof < 1:
        print(f"DoF is lower in {ndof_2} compared to {ndof_1}")

        return np.nan, np.nan, n_dof

    p_value = chi2.sf((stat_1 - stat_2), n_dof)
    gaussian_sigmas = norm.isf(p_value / 2)

    if not np.isfinite(gaussian_sigmas):
        gaussian_sigmas = np.sqrt((stat_1 - stat_2))

    return p_value, gaussian_sigmas, n_dof


def check_model_preference_aic(list_wstat, list_dof):
    """
    Akaike Information Criterion (AIC) preference over a list of wstat and DoF
    (degree of freedom) to get relative likelihood of a given list of best-fit
    models.

    Parameters
    ----------
    list_wstat: list
        List of wstat or -2 Log likelihood values for a list of models.
    list_dof: list
        List of degrees of freedom or list of free parameters, for a list of models.

    Returns
    -------
    list_p: list
        List of relative likelihood probabilities, for a list of models.
    """
    list_aic = []
    for w, d in zip(list_wstat, list_dof):
        aic = 2 * w + 2 * d
        list_aic.append(aic)
    list_aic = np.array(list_aic)

    aic_min = np.min(list_aic)

    list_b = []
    for a in list_aic:
        b = np.exp((aic_min - a) / 2)
        list_b.append(b)
    list_b = np.array(list_b)

    list_p = []
    for bb in list_b:
        bbb = bb / np.sum(list_b)
        list_p.append(bbb)
    list_p = np.array(list_p)

    return list_p
