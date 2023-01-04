"""
Config-driven high level analysis interface.
"""
import logging

from gammapy.datasets import Datasets

from asgardpy.config.generator import AsgardpyConfig
from asgardpy.data.base import AnalysisStep
from asgardpy.data.target import set_models

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
        """
        Set a given Model to the final datasets object.
        """
        self.datasets = set_models(
            config=self.config.target, datasets=self.datasets, models=models, extend=False
        )

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
                """
                print(type(datasets_list), datasets_list)
                check = datasets_list.models == None
                print(check)
                if datasets_list.models == None:
                    print("Trying to add the config based model")
                    datasets_list = set_models(
                        config=self.config.target,
                        datasets=datasets_list
                    )
                """
                for data in datasets_list:
                    print(data.models)
                    """
                    #print("New model:", data)
                    data = set_models(
                        config=self.config.target,
                        datasets=data
                    )
                    """
                    self.datasets.append(data)
            else:
                # Running DL4 functions on a given Datasets object
                print(self.datasets.models)
                # print(self.datasets.models)
                analysis_step = AnalysisStep.create(step, self.config, **kwargs)
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
