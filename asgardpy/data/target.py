"""
Classes containing the Target config parameters for the high-level interface and
also the functions involving Models generation and assignment to datasets.
"""

from pathlib import Path
from typing import List

from astropy.coordinates import SkyCoord
from gammapy.maps import Map
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.models import (
    SPATIAL_MODEL_REGISTRY,
    SPECTRAL_MODEL_REGISTRY,
    DatasetModels,
    EBLAbsorptionNormSpectralModel,
    LogParabolaSpectralModel,
    Models,
    PointSpatialModel,
    PowerLawNormSpectralModel,
    PowerLawSpectralModel,
    SkyModel,
    TemplateSpatialModel,
    create_fermi_isotropic_diffuse_model,
)

from asgardpy.data.base import BaseConfig
from asgardpy.data.geom import SkyCoordConfig

__all__ = [
    "EBLAbsorptionModel",
    "SpectralModelConfig",
    "SpatialModelConfig",
    "Target",
    "set_models",
    "config_to_dict",
    "read_models_from_asgardpy_config",
    "xml_to_gammapy_model_params",
    "create_source_skymodel",
    "create_iso_diffuse_skymodel",
    "create_gal_diffuse_skymodel",
]


# Basic components to define the Target Config and any Models Config
class EBLAbsorptionModel(BaseConfig):
    model_name: str = "dominguez"
    type: str = "EBLAbsorptionNormSpectralModel"
    redshift: float = 0.4
    alpha_norm: float = 1.0


class ModelParams(BaseConfig):
    name: str = ""
    value: float = 1
    unit: str = " "
    error: float = 0.1
    min: float = 0.1
    max: float = 10
    frozen: bool = True


class SpectralModelConfig(BaseConfig):
    model_name: str = ""
    type: str = ""
    parameters: List[ModelParams] = [ModelParams()]
    ebl_abs: EBLAbsorptionModel = EBLAbsorptionModel()


class SpatialModelConfig(BaseConfig):
    model_name: str = ""
    type: str = ""
    parameters: List[ModelParams] = [ModelParams()]


class SkyModelComponent(BaseConfig):
    name: str = ""
    type: str = "SkyModel"
    spectral: SpectralModelConfig = SpectralModelConfig()
    spatial: SpatialModelConfig = SpatialModelConfig()


class Target(BaseConfig):
    source_name: str = ""
    sky_position: SkyCoordConfig = SkyCoordConfig()
    use_uniform_position: bool = True
    models_file: Path = Path(".")
    extended: bool = False
    components: SkyModelComponent = SkyModelComponent()
    covariance: str = ""
    from_fermi: bool = False


# Function for Models assignment
def set_models(
    config, datasets, datasets_name_list=None, models=None, target_source_name=None, extend=False
):
    """
    Set models on given Datasets.

    Parameters
    ----------
    config: `AsgardpyConfig` or others?
        AsgardpyConfig containing target information.
    datasets: `gammapy.datasets.Datasets`
        Datasets object
    dataset_name_list: List
        List of datasets_names to be used on the Models, before assigning them
        to the given datasets.
    models : `~gammapy.modeling.models.Models` or str
        Models object or YAML models string
    target_source_name: str
        Name of the Target source, to use to update only that Model's
        datasets_names, when a list of more than 1 models are provided.
    extend : bool
        Extend the existing models on the datasets or replace them with
        another model, maybe a Background Model. Not worked out currently.

    Returns
    ------
    datasets: `gammapy.datasets.Datasets`
        Datasets object with Models assigned.
    """
    # Have some checks on argument types
    if isinstance(models, DatasetModels) or isinstance(models, list):
        models = Models(models)
    elif config.components:
        spectral_model, spatial_model = read_models_from_asgardpy_config(config)
        models = Models(
            SkyModel(
                spectral_model=spectral_model,
                spatial_model=spatial_model,
                name=config.source_name,
            )
        )
    elif isinstance(models, str):  # Check this condition
        models = Models.from_yaml(models)
    else:
        raise TypeError(f"Invalid type: {type(models)}")

    # if extend:
    # For extending a Background Model
    #    Models(models).extend(self.bkg_models)

    if datasets_name_list is None:
        datasets_name_list = datasets.names

    if target_source_name is None:
        target_source_name = config.source_name

    if len(models) > 1:
        models[target_source_name].datasets_names = datasets_name_list
    else:
        models.datasets_names = datasets_name_list

    datasets.models = models

    return datasets


# Functions for Models generation
def read_models_from_asgardpy_config(config):
    """
    Reading Models information from AsgardpyConfig and return Spectral and
    Spatial Models object to be combined later into SkyModels/Models object.

    Parameters
    ----------
    config: `AsgardpyConfig`
        Config section containing Target source information

    Returns
    -------
    spectral_model: `gammapy.modeling.models.SpectralModel`
        Spectral Model components of a gammapy SkyModel object.
    spatial_model: `gammapy.modeling.models.SpatialModel`
        Spatial Model components of a gammapy SkyModel object.
    """
    model_config = config.components

    # Spectral Model
    if model_config.spectral.ebl_abs.model_name is not None:
        model1 = SPECTRAL_MODEL_REGISTRY.get_cls(model_config.spectral.type)().from_dict(
            {"spectral": config_to_dict(model_config.spectral)}
        )

        ebl_model = model_config.spectral.ebl_abs
        model2 = EBLAbsorptionNormSpectralModel.read_builtin(
            ebl_model.model_name, redshift=ebl_model.redshift
        )
        if ebl_model.alpha_norm:
            model2.alpha_norm.value = ebl_model.alpha_norm
        spectral_model = model1 * model2
    else:
        spectral_model = SPECTRAL_MODEL_REGISTRY.get_cls(model_config.spectral.type)().from_dict(
            {"spectral": config_to_dict(model_config.spectral)}
        )
    spectral_model.name = config.source_name

    # Spatial model if provided
    if model_config.spatial.model_name is not None:
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

    Return
    ------
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


def xml_to_gammapy_model_params(params, is_target=False, keep_sign=False, lp_is_intrinsic=False):
    """
    Convert the Models information from XML model of FermiTools to Gammapy
    standards and return Parameters list.
    Details of the XML model can be seen at
    https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html
    and with examples at
    https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/xml_model_defs.html

    Parameters
    ----------
    params: `gammapy.modeling.Parameters`
        List of gammapy Parameter object of a particular Model
    is_target: bool
        Boolean to check if the given list of Parameters belong to the target
        source model or not.
    keep_sign: bool
        Boolean to keep the same sign on the parameter values or not.
    lp_is_intrinsic: bool
        Boolean to check if the given Model assumes an intrinsic spectrum of
        Log Parabola type, but uses Power Law and then adds EBL attenuation
        for characterizing the curvature.

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
                new_par["frozen"] = (par[key_] == "0") and not is_target
            new_par["unit"] = ""
            new_par["is_norm"] = False

            # Using the nomenclature as used in Gammapy
            # Making scale = 1, by multiplying it to the value, min and max
            if par["@name"].lower() in ["norm", "prefactor", "integral"]:
                new_par["name"] = "amplitude"
                new_par["unit"] = "cm-2 s-1 TeV-1"
                new_par["is_norm"] = True
            if par["@name"].lower() in ["scale", "eb"]:
                new_par["name"] = "reference"
            if par["@name"].lower() in ["breakvalue"]:
                new_par["name"] = "ebreak"
            if par["@name"].lower() in ["lowerlimit"]:
                new_par["name"] = "emin"
            if par["@name"].lower() in ["upperlimit"]:
                new_par["name"] = "emax"
            if par["@name"].lower() in ["cutoff"]:
                new_par["name"] = "lambda_"
                new_par["value"] = 1.0 / new_par["value"]
                new_par["min"] = 1.0 / new_par["max"]
                new_par["max"] = 1.0 / new_par["min"]
                new_par["unit"] = "TeV-1"
            if par["@name"].lower() in ["index"]:
                new_par["name"] = "index"

        # Some modifications:
        if new_par["name"] in ["reference", "ebreak", "emin", "emax"]:
            new_par["unit"] = "TeV"
            new_par["value"] = float(new_par["value"]) * float(new_par["scale"]) * 1e-6
            new_par["min"] = float(new_par["min"]) * float(new_par["scale"]) * 1e-6
            new_par["max"] = float(new_par["max"]) * float(new_par["scale"]) * 1e-6
        if new_par["name"] in ["amplitude"]:
            new_par["value"] = float(new_par["value"]) * float(new_par["scale"]) * 1e6
            new_par["min"] = float(new_par["min"]) * float(new_par["scale"]) * 1e6
            new_par["max"] = float(new_par["max"]) * float(new_par["scale"]) * 1e6
        if new_par["name"] == "index" and not keep_sign:
            # Other than EBL Attenuated Power Law
            new_par["value"] = -1 * float(new_par["value"])

            # Reverse the limits while changing the sign
            min_ = float(new_par["min"])
            max_ = float(new_par["max"])
            new_par["min"] = -1 * max_
            new_par["max"] = -1 * min_

        new_par["error"] = 0
        new_param = Parameter(name=new_par["name"], value=new_par["value"])
        new_param.min = new_par["min"]
        new_param.max = new_par["max"]
        new_param.unit = new_par["unit"]
        new_param.frozen = new_par["frozen"]
        new_param._is_norm = new_par["is_norm"]

        new_params.append(new_param)

    params_final2 = Parameters(new_params)

    return params_final2


def create_source_skymodel(config_target, source, aux_path, lp_is_intrinsic=False):
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
    lp_is_intrinsic: bool
        Boolean to check if the given Model assumes an intrinsic spectrum of
        Log Parabola type, but uses Power Law and then adds EBL attenuation
        for characterizing the curvature.

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
    spatial_pars = source["spatialModel"]["parameter"]

    source_name_check = source_name.replace("_", "").replace(" ", "")
    target_check = config_target.source_name.replace("_", "").replace(" ", "")

    # initialized to check for the case if target spectral model information
    # is to be taken from the Config
    spectral_model = None

    # Check if target_source file exists
    is_source_target = False
    ebl_atten_pl = False

    # If Target source model's spectral component is to be taken from Config
    # and not from Fermi.
    if source_name_check == target_check:
        source_name = config_target.source_name
        is_source_target = True

        # Only taking the spectral model information right now.
        if not config_target.from_fermi:
            spectral_model, _ = read_models_from_asgardpy_config(config_target)

    if spectral_model is None:
        # Define the Spectral Model type for Gammapy
        for spec in spectrum:
            if spec["@name"] not in ["GalDiffModel", "IsoDiffModel"]:
                if spectrum_type == "PLSuperExpCutoff":
                    spectrum_type_final = "ExpCutoffPowerLawSpectralModel"
                elif spectrum_type == "PLSuperExpCutoff4":
                    spectrum_type_final = "SuperExpCutoffPowerLaw4FGLDR3SpectralModel"
                else:
                    spectrum_type_final = f"{spectrum_type}SpectralModel"

                spectral_model = SPECTRAL_MODEL_REGISTRY.get_cls(spectrum_type_final)()

                if spectrum_type == "LogParabola" and "EblAtten" in source["spectrum"]["@type"]:
                    if lp_is_intrinsic:
                        spectral_model = LogParabolaSpectralModel()
                    else:
                        ebl_atten_pl = True
                        spectral_model = PowerLawSpectralModel()

        # Read the parameter values from XML file to create SpectralModel
        params_list = xml_to_gammapy_model_params(
            spectrum,
            is_target=is_source_target,
            keep_sign=ebl_atten_pl,
            lp_is_intrinsic=lp_is_intrinsic,
        )

        for param_ in params_list:
            setattr(spectral_model, param_.name, param_)
        config_spectral = config_target.components.spectral
        ebl_absorption_included = config_spectral.ebl_abs is not None

        if is_source_target and ebl_absorption_included:
            ebl_absorption = config_spectral.ebl_abs
            ebl_model = ebl_absorption.model_name
            ebl_spectral_model = EBLAbsorptionNormSpectralModel.read_builtin(
                ebl_model, redshift=ebl_absorption.redshift
            )
            spectral_model = spectral_model * ebl_spectral_model

    # Reading Spatial model from the XML file
    if source["spatialModel"]["@type"] == "SkyDirFunction":
        fk5_frame = SkyCoord(
            f"{spatial_pars[0]['@value']} deg",
            f"{spatial_pars[1]['@value']} deg",
            frame="fk5",
        )
        gal_frame = fk5_frame.transform_to("galactic")
        spatial_model = PointSpatialModel.from_position(gal_frame)
    elif source["spatialModel"]["@type"] == "SpatialMap":
        file_name = source["spatialModel"]["@file"].split("/")[-1]
        file_path = aux_path / f"Templates/{file_name}"

        spatial_map = Map.read(file_path)
        spatial_map = spatial_map.copy(unit="sr^-1")

        spatial_model = TemplateSpatialModel(spatial_map, filename=file_path)

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
    name as per the observation key.
    """
    diff_iso = create_fermi_isotropic_diffuse_model(
        filename=iso_file, interp_kwargs={"fill_value": None}
    )
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
    template_diffuse = TemplateSpatialModel(diff_gal, normalize=False)
    source = SkyModel(
        spectral_model=PowerLawNormSpectralModel(),
        spatial_model=template_diffuse,
        name="diffuse-iem",
    )
    source.parameters["norm"].min = 0
    source.parameters["norm"].max = 10
    source.parameters["norm"].frozen = False

    return source
