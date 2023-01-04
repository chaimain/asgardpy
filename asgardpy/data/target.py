"""
Classes containing the Target config parameters for the high-level interface.
Also contains dome functions for setting various SkyModels for datasets.
"""

from pathlib import Path
from typing import List

from astropy.coordinates import SkyCoord
from gammapy.maps import Map
from gammapy.modeling import Parameters  # , Parameter
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
    "xml_to_gammapy_model_params",
    "create_source_skymodel",
    "create_iso_diffuse_skymodel",
    "create_gal_diffuse_skymodel",
]


class EBLAbsorptionModel(BaseConfig):
    model_name: str = "dominguez"
    type: str = "EBLAbsorptionNormSpectralModel"
    redshift: float = 0.4
    alpha_norm: float = 1.0


class ModelParams(BaseConfig):
    name: str = None
    value: float = None
    unit: str = None
    error: float = None
    min: float = None
    max: float = None
    frozen: bool = True


class SpectralModelConfig(BaseConfig):
    model_name: str = None
    type: str = None
    parameters: List[ModelParams] = [ModelParams()]
    ebl_abs: EBLAbsorptionModel = EBLAbsorptionModel()


class SpatialModelConfig(BaseConfig):
    model_name: str = None
    type: str = None
    parameters: List[ModelParams] = [ModelParams()]


class SkyModelComponent(BaseConfig):
    name: str = None
    type: str = "SkyModel"
    spectral: SpectralModelConfig = SpectralModelConfig()
    spatial: SpatialModelConfig = SpatialModelConfig()


class Target(BaseConfig):
    source_name: str = None
    sky_position: SkyCoordConfig = SkyCoordConfig()
    use_uniform_position: bool = True
    models_file: Path = None
    extended: bool = False
    components: SkyModelComponent = SkyModelComponent()
    covariance: str = None
    from_fermi: bool = False


def set_models(config, datasets, models=None, extend=False):
    """
    Set models on given Datasets.

    Parameters
    ----------
    config: AsgardpyConfig containing target information.
    datasets: Dataset object or Datasets?
    models : `~gammapy.modeling.models.Models` or str
        Models object or YAML models string
    extend : bool
        Extend the existing models on the datasets or replace them with
        another model, maybe a Background Model. Not worked out currently.
    """
    # Have some checks on argument types
    if config.components:
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
            spec_model = model1 * model2
        else:
            spec_model = SPECTRAL_MODEL_REGISTRY.get_cls(model_config.spectral.type)().from_dict(
                {"spectral": config_to_dict(model_config.spectral)}
            )
        spec_model.name = config.source_name
        # Spatial model if provided
        if model_config.spatial.model_name is not None:
            spat_model = SPATIAL_MODEL_REGISTRY.get_cls(model_config.spatial.type)().from_dict(
                {"spatial": config_to_dict(model_config.spatial)}
            )
        else:
            spat_model = None
        # Final SkyModel
        models = Models(
            SkyModel(
                spectral_model=spec_model,
                spatial_model=spat_model,
                name=config.source_name,
            )
        )
    elif isinstance(models, str):  # Check this condition
        models = Models.from_yaml(models)
    elif isinstance(models, Models):
        pass
    elif isinstance(models, DatasetModels) or isinstance(models, list):  # Essential?
        models = Models(models)
    else:
        raise TypeError(f"Invalid type: {models!r}")

    # if extend:
    # For extending a Background Model
    #    Models(models).extend(self.bkg_models)

    datasets.models = models

    return datasets


def config_to_dict(model_config):
    """
    Convert the Spectral/Spatial models into dict.
    Probably an extra step and maybe removed later.
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


def xml_to_gammapy_model_params(
    params, spectrum_type, is_target=False, keep_sign=False, lp_is_intrinsic=False
):
    """
    Make some basic conversions on the spectral model parameters
    """
    params_list = []
    for par in params:
        new_par = {}
        # For EBL Attenuated Power Law, it is taken with LogParabola model
        # and turning beta value off
        if lp_is_intrinsic and par["@name"] == "beta":
            continue

        for k in par.keys():
            # Replacing the "@par_name" information of each parameter without the "@"
            if k != "@free":
                new_par[k[1:].lower()] = par[k]
            else:
                new_par["frozen"] = (par[k] == "0") and not is_target
            new_par["unit"] = ""
            new_par["is_norm"] = False

            # Using the nomenclature as used in Gammapy
            # Make scale = 1, by multiplying it to the value, min and max?
            if par["@name"].lower() in ["norm", "prefactor", "integral"]:
                new_par["name"] = "amplitude"
                new_par["unit"] = "cm-2 s-1 MeV-1"
                new_par["is_norm"] = True
            if par["@name"].lower() in ["scale", "eb"]:
                new_par["name"] = "reference"
                new_par["frozen"] = par[k] == "0"
            if par["@name"].lower() in ["breakvalue"]:
                new_par["name"] = "ebreak"
            if par["@name"].lower() in ["lowerlimit"]:
                new_par["name"] = "emin"
            if par["@name"].lower() in ["upperlimit"]:
                new_par["name"] = "emax"
            if par["@name"].lower() in ["cutoff"]:
                new_par["name"] = "lambda_"
                new_par["value"] = 1.0 / new_par["value"]
                new_par["min"] = 1.0 / new_par["min"]
                new_par["max"] = 1.0 / new_par["max"]
                new_par["unit"] = "MeV-1"

        # More modifications:
        if new_par["name"] in ["reference", "ebreak", "emin", "emax"]:
            new_par["unit"] = "MeV"
        if new_par["name"] == "index" and not keep_sign:
            # Other than EBL Attenuated Power Law
            new_par["value"] *= -1
            new_par["min"] *= -1
            new_par["max"] *= -1
        new_par["error"] = 0
        params_list.append(new_par)

    params_final = Parameters.from_dict(params_list)

    return params_final


def create_source_skymodel(config_target, src, aux_path, lp_is_intrinsic=False):
    """
    Build SkyModels from a given AsgardpyConfig section of the target
    source information, list of LAT files and other relevant information.
    """
    source_name = src["@name"]
    spectrum_type = src["spectrum"]["@type"].split("EblAtten::")[-1]
    spectrum = src["spectrum"]["parameter"]
    spatial_pars = src["spatialModel"]["parameter"]

    source_name_red = source_name.replace("_", "").replace(" ", "")
    target_red = config_target.source_name.replace("_", "").replace(" ", "")

    # Check if target_source file exists
    if source_name_red == target_red:
        source_name = config_target.source_name
        is_src_target = True
        # self.log.debug("Detected target source")
    else:
        is_src_target = False

    for spec in spectrum:
        if spec["@name"] not in ["GalDiffModel", "IsoDiffModel"]:
            if spectrum_type == "PLSuperExpCutoff":
                spectrum_type_final = "ExpCutoffPowerLawSpectralModel"
            elif spectrum_type == "PLSuperExpCutoff4":
                spectrum_type_final = "SuperExpCutoffPowerLaw4FGLDR3SpectralModel"
            else:
                spectrum_type_final = f"{spectrum_type}SpectralModel"

            spec_model = SPECTRAL_MODEL_REGISTRY.get_cls(spectrum_type_final)()
            # spec_model.name = source_name
            ebl_atten_pl = False

            if spectrum_type == "LogParabola" and "EblAtten" in src["spectrum"]["@type"]:
                if lp_is_intrinsic:
                    spec_model = LogParabolaSpectralModel()
                else:
                    ebl_atten_pl = True
                    spec_model = PowerLawSpectralModel()

    params_list = xml_to_gammapy_model_params(
        spectrum,
        spectrum_type,
        is_target=is_src_target,
        keep_sign=ebl_atten_pl,
        lp_is_intrinsic=lp_is_intrinsic,
    )
    spec_model.from_parameters(params_list)
    config_spectral = config_target.components.spectral
    ebl_absorption_included = config_spectral.ebl_abs is not None

    if is_src_target and ebl_absorption_included:
        ebl_absorption = config_spectral.ebl_abs
        ebl_model = ebl_absorption.model_name
        ebl_spectral_model = EBLAbsorptionNormSpectralModel.read_builtin(
            ebl_model, redshift=ebl_absorption.redshift
        )
        spec_model = spec_model * ebl_spectral_model

    if src["spatialModel"]["@type"] == "SkyDirFunction":
        fk5_frame = SkyCoord(
            f"{spatial_pars[0]['@value']} deg",
            f"{spatial_pars[1]['@value']} deg",
            frame="fk5",
        )
        gal_frame = fk5_frame.transform_to("galactic")
        spatial_model = PointSpatialModel.from_position(gal_frame)
    elif src["spatialModel"]["@type"] == "SpatialMap":
        file_name = src["spatialModel"]["@file"].split("/")[-1]
        file_path = aux_path / f"Templates/{file_name}"

        spatial_map = Map.read(file_path)
        spatial_map = spatial_map.copy(unit="sr^-1")

        spatial_model = TemplateSpatialModel(spatial_map, filename=file_path)

    spatial_model.freeze()
    source_sky_model = SkyModel(
        spectral_model=spec_model,
        spatial_model=spatial_model,
        name=source_name,
    )

    return source_sky_model, is_src_target


def create_iso_diffuse_skymodel(iso_file, key):
    """
    Create a SkyModel of the Fermi Isotropic Diffuse Model.
    """
    diff_iso = create_fermi_isotropic_diffuse_model(
        filename=iso_file, interp_kwargs={"fill_value": None}
    )
    diff_iso._name = f"{diff_iso.name}-{key}"

    # Parameters' limits generalization?
    diff_iso.spectral_model.model1.parameters[0].min = 0.001
    diff_iso.spectral_model.model1.parameters[0].max = 10
    diff_iso.spectral_model.model2.parameters[0].min = 0
    diff_iso.spectral_model.model2.parameters[0].max = 10

    return diff_iso


def create_gal_diffuse_skymodel(diff_gal):
    """
    Create SkyModel of the Diffuse Galactic sources.
    Maybe a repeat of code from the _cutout function.
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
