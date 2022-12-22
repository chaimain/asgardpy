"""
Classes containing the DL4 products config parameters for the high-level interface
"""
from enum import Enum
from asgardpy.config.generator import AsgardpyConfig
from gammapy.datasets import Datasets
from gammapy.modeling import Fit

from asgardpy.data.base import AnalysisStep, AngleType, BaseConfig, EnergyRangeConfig, TimeRangeConfig
from asgardpy.data.geom import EnergyAxisConfig

__all__ = [
    "FluxPointsConfig",
    "LightCurveConfig",
    "FitConfig",
    "ExcessMapConfig",
    "AsgardpyAnalysis",
]


class FluxPointsConfig(BaseConfig):
    energy: EnergyAxisConfig = EnergyAxisConfig()
    source: str = "source"
    parameters: dict = {"selection_optional": "all"}


class LightCurveConfig(BaseConfig):
    time_intervals: TimeRangeConfig = TimeRangeConfig()
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()
    source: str = "source"
    parameters: dict = {"selection_optional": "all"}


class BackendEnum(str, Enum):
    minuit = "minuit"
    scipy = "scipy"


class FitConfig(BaseConfig):
    fit_range: EnergyRangeConfig = EnergyRangeConfig()
    backend: BackendEnum = None
    optimize_opts: dict = {}
    covariance_opts: dict = {}
    confidence_opts: dict = {}
    store_trace: bool = True


class ExcessMapConfig(BaseConfig):
    correlation_radius: AngleType = "0.1 deg"
    parameters: dict = {}
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()


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
        self.datasets = None
        self.fit = Fit()
        self.fit_result = None
        self.flux_points = None

    @property
    def models(self):
        if not self.datasets:
            raise RuntimeError("No datasets defined. Impossible to set models.")
        return self.datasets.models

    @models.setter
    def models(self, models):
        self.set_models(models, extend=False)

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
        for k, step in enumerate(steps):
            analysis_step = AnalysisStep.create(
                step, self.config, log=self.log, overwrite=overwrite, **kwargs
            )
            analysis_step.run(self)

    # keep these methods to be backward compatible
    def get_1d_dataset(self):
        """Produce stacked 1D datasets."""
        self.run(steps=["dataset-1d-datasets"])

    def get_3d_datasets(self):
        """Produce stacked 3D datasets."""
        self.run(steps=["dataset-3d-datasets"])

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

    def read_datasets(self): ## Necessary?
        """
        Read datasets from YAML file.
        File names are taken from the configuration file.

        """

        filename = self.config.general.datasets_file
        filename_models = self.config.general.models_file
        if filename is not None:
            self.datasets = Datasets.read(filename)
            self.log.info(f"Datasets loaded from {filename}.")
            if filename_models is not None:
                self.read_models(filename_models, extend=False)
        else:
            raise RuntimeError("Missing datasets_file in config.general")

    def write_datasets(self, overwrite=True, write_covariance=True):
        """Write datasets to YAML file.
        File names are taken from the configuration file.

        Parameters
        ----------
        overwrite : bool
            overwrite datasets FITS files
        write_covariance : bool
            save covariance or not
        """

        filename = self.config.general.datasets_file
        filename_models = self.config.general.models_file
        if filename is not None:
            self.datasets.write(
                filename,
                filename_models,
                overwrite=overwrite,
                write_covariance=write_covariance,
            )
            self.log.info(f"Datasets stored to {filename}.")
            self.log.info(f"Datasets stored to {filename_models}.")
        else:
            raise RuntimeError("Missing datasets_file in config.general")
