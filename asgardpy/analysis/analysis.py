"""
Config-driven high level analysis interface.
"""
import logging

from gammapy.datasets import Datasets
from gammapy.modeling.models import (
    DatasetModels,
    EBLAbsorptionNormSpectralModel,
    Models,
    SkyModel,
    SpatialModel,
    SpectralModel,
    SPECTRAL_MODEL_REGISTRY,
    SPATIAL_MODEL_REGISTRY
)

from asgardpy.config.generator import AsgardpyConfig
from asgardpy.data.base import AnalysisStep

log = logging.getLogger(__name__)

__all__ = ["AsgardpyAnalysis"]


class AsgardpyAnalysis:
    """
    Config-driven high level analysis interface.

    It is initialized by default with a set of configuration parameters and values declared in
    an internal high level interface model, though the user can also provide configuration
    parameters passed as a nested dictionary at the moment of instantiation. In that case these
    parameters will overwrite the default values of those present in the configuration file.

    Parameters
    ----------
    config : dict or `AsgardpyConfig`
        Configuration options following `AsgardpyConfig` schema
    """

    def __init__(self, config):
        self.log = log
        self.config = config
        self.config.set_logging()
        self.datasets = Datasets()
        # self.fit = Fit()
        # self.fit_result = None
        # self.flux_points = None

    @property
    def models(self):
        if not self.datasets:
            raise RuntimeError("No datasets defined. Impossible to set models.")
        return self.datasets.models

    @models.setter
    def models(self, models):
        self.set_models(models, extend=False)

    def set_models(self, models=None, extend=False):
        """
        Set models on Datasets.
        Parameters
        ----------
        models : `~gammapy.modeling.models.Models` or str
            Models object or YAML models string
        extend : bool
            Extend the existing models on the datasets or replace them with
            another model, maybe a Background Model. Not worked out currently.
        """
        if self.config.target.components:
            model_config = self.config.target.components
            # Spectral Model
            if model_config.spectral.ebl_abs.model_name is not None:
                model1 = SPECTRAL_MODEL_REGISTRY.get_cls(
                    model_config.spectral.type
                )().from_dict(
                    {"spectral": self.config_to_dict(model_config.spectral)}
                )

                ebl_model = model_config.spectral.ebl_abs
                model2 = EBLAbsorptionNormSpectralModel.read_builtin(
                    ebl_model.model_name, redshift=ebl_model.redshift
                )
                if ebl_model.alpha_norm:
                    model2.alpha_norm.value = ebl_model.alpha_norm
                spec_model = model1 * model2
            else:
                spec_model = SPECTRAL_MODEL_REGISTRY.get_cls(
                    model_config.spectral.type
                )().from_dict(
                    {"spectral": self.config_to_dict(model_config.spectral)}
                )
            # Spatial model if provided
            if model_config.spatial.model_name is not None:
                spat_model = SPATIAL_MODEL_REGISTRY.get_cls(
                    model_config.spatial.type
                )().from_dict(
                    {"spatial": self.config_to_dict(model_config.spatial)}
                )
            else:
                spat_model = None
            # Final SkyModel
            models = SkyModel(
                spectral_model=spec_model,
                spatial_model=spat_model,
                name=self.config.target.source_name,
            )
        elif isinstance(models, str):  # Check this condition
            models = Models.from_yaml(models)
        elif isinstance(models, Models):
            pass
        elif isinstance(models, DatasetModels) or isinstance(models, list):  # Essential?
            models = Models(models)
        else:
            raise TypeError(f"Invalid type: {models!r}")

        #if extend:
        # For extending a Background Model
        #    Models(models).extend(self.bkg_models)

        self.datasets.models = models

    def config_to_dict(self, model_config):
        """
        Convert the Spectral/Spatial models defined in asgardpy/data/target.py
        into dict.
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

    @property
    def config(self):
        """
        Analysis configuration (`AsgardpyConfig`)
        """
        return self._config

    @config.setter
    def config(self, value):
        if isinstance(value, dict):
            self._config = AsgardpyConfig(**value)
        elif isinstance(value, AsgardpyConfig):
            self._config = value
        else:
            raise TypeError("config must be dict or AsgardpyConfig.")

    def run(self, steps=None, overwrite=None, **kwargs):
        if steps is None:
            steps = self.config.general.steps
            overwrite = self.config.general.overwrite
        else:
            if overwrite is None:
                overwrite = True

        for step in steps:
            # Always start with 3D datasets. Probably add a check or fail-safe
            if "datasets" in step:
                analysis_step = AnalysisStep.create(step, self.config, **kwargs)
                datasets_list = analysis_step.run()

                # Add to the final list of datasets
                for data in datasets_list:
                    self.datasets.append(data)
                self.set_models()
            else:
                # Running DL4 functions on a given Datasets object
                analysis_step = AnalysisStep.create(
                    step, self.config, **kwargs
                )
                analysis_step.run(datasets=self.datasets)

    # keep these methods to be backward compatible
    def get_1d_dataset(self):
        """Produce stacked 1D datasets."""
        self.run(steps=["datasets-1d"])

    def get_3d_datasets(self):
        """Produce stacked 3D datasets."""
        self.run(steps=["datasets-3d"])

    def run_fit(self):
        """Fitting reduced datasets to model."""
        self.run(steps=["fit"])

    def get_flux_points(self):
        """Calculate flux points for a specific model component."""
        self.run(steps=["flux-points"])

    def get_excess_map(self):
        """Calculate excess map with respect to the current model."""
        self.run(steps=["excess-map"])

    def get_light_curve(self):
        """Calculate light curve for a specific model component."""
        self.run(steps=["light-curve"])

    def update_config(self, config):
        self.config = self.config.update(config=config)
