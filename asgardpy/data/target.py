"""
Classes containing the Target config parameters for the high-level interface
"""

from pathlib import Path
from typing import List

from gammapy.modeling.models import (
    SPATIAL_MODEL_REGISTRY,
    SPECTRAL_MODEL_REGISTRY,
    DatasetModels,
    EBLAbsorptionNormSpectralModel,
    Models,
    SkyModel
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
