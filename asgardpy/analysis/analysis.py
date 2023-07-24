"""
Config-driven high level analysis interface.
"""
import logging

import numpy as np
from gammapy.datasets import Datasets
from gammapy.modeling.models import Models
from gammapy.stats import CashCountsStatistic, WStatCountsStatistic

from asgardpy.base import AnalysisStep
from asgardpy.config import AsgardpyConfig
from asgardpy.data import get_goodness_of_fit_stats, set_models

log = logging.getLogger(__name__)

__all__ = ["AsgardpyAnalysis"]


class AsgardpyAnalysis:
    """
    Config-driven high level analysis interface.

    It is initialized by default with a set of configuration parameters and
    values declared in an internal high level interface model, though the user
    can also provide configuration parameters passed as a nested dictionary at
    the moment of instantiation. In that case these parameters will overwrite
    the default values of those present in the configuration file.

    A specific example of an upgrade of the configuration parameters is when
    the Target Models information is provided as a path to a separate yaml file,
    which is readable with AsgardpyConfig. In this case, the configuration used
    in AsgardpyAnalysis is updated in the initialization step itself.

    Parameters
    ----------
    config : dict or `AsgardpyConfig`
        Configuration options following `AsgardpyConfig` schema
    """

    def __init__(self, config):
        self.log = log
        self.config = config

        if self.config.target.models_file.is_file():
            other_config = AsgardpyConfig.read(self.config.target.models_file)
            self.config = self.config.update(other_config)

        self.config.set_logging()
        self.datasets = Datasets()
        self.instrument_spectral_info = {
            "name": [],
            "spectral_energy_ranges": [],
            "DoF": 0,
            "TS_H0": 0,
        }
        self.dataset_name_list = []

        self.final_model = Models()
        self.final_data_products = ["fit", "fit_result", "flux_points"]

        for data_product in self.final_data_products:
            setattr(self, data_product, None)

    @property
    def models(self):
        """
        Display the assigned Models.
        """
        if not self.datasets:
            raise RuntimeError("No datasets defined. Impossible to set models.")
        return self.datasets.models

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
        """
        Main function to run the AnalaysisSteps provided.
        """
        if steps is None:
            steps = self.config.general.steps
            overwrite = self.config.general.overwrite
        else:
            if overwrite is None:
                overwrite = True

        dl3_dl4_steps = [step for step in steps if "datasets" in step]
        dl4_dl5_steps = [step for step in steps if "datasets" not in step]

        if len(dl3_dl4_steps) > 0:
            self.log.info("Perform DL3 to DL4 process!")

            for step in dl3_dl4_steps:
                analysis_step = AnalysisStep.create(step, self.config, **kwargs)

                datasets_list, models_list, instrument_spectral_info = analysis_step.run()

                if models_list:
                    # This step is only valid for 3D Datasets which have a list of models
                    target_source_model = models_list[self.config.target.source_name]

                    if target_source_model.spatial_model:
                        # If the target source has Spatial model included,
                        # only then (?) get all the models as final_model.
                        # Needs reconsideration.
                        for model_ in models_list:
                            self.final_model.append(model_)
                    else:
                        self.log.info(
                            "The target source %s only has spectral model",
                            self.config.target.source_name,
                        )

                # To get all datasets_names from the datasets and update the final datasets list
                for data in datasets_list:
                    if data.name not in self.dataset_name_list:
                        self.dataset_name_list.append(data.name)
                    self.datasets.append(data)

                # Update the name, DoF and spectral energy ranges for each
                # instrument Datasets, to be used for the DL4 to DL5 processes.
                for name in instrument_spectral_info["name"]:
                    self.instrument_spectral_info["name"].append(name)

                for edges in instrument_spectral_info["spectral_energy_ranges"]:
                    self.instrument_spectral_info["spectral_energy_ranges"].append(edges)

                self.instrument_spectral_info["DoF"] += instrument_spectral_info["DoF"]

        self.datasets, self.final_model = set_models(
            self.config.target,
            self.datasets,
            self.dataset_name_list,
            models=self.final_model,
        )

        # Subtract the total number of free model parameters from the total
        # number of degrees of freedom
        n_free_params = len(list(self.final_model.parameters.free_parameters))
        self.instrument_spectral_info["DoF"] -= n_free_params

        # Evaluate the TS for the null hypothesis
        for data in self.datasets:
            region = data.counts.geom.center_skydir

            if data.stat_type == "cash":
                counts_on = (data.counts.copy() * data.mask).get_spectrum(region).data
                mu_on = (data.npred() * data.mask).get_spectrum(region).data

                stat = CashCountsStatistic(
                    counts_on,
                    mu_on,
                )
            elif data.stat_type == "wstat":
                counts_on = (data.counts.copy() * data.mask).get_spectrum(region)
                counts_off = np.nan_to_num((data.counts_off * data.mask).get_spectrum(region))
                alpha = (
                    np.nan_to_num((data.background * data.mask).get_spectrum(region)) / counts_off
                )
                mu_signal = np.nan_to_num((data.npred_signal() * data.mask).get_spectrum(region))

                stat = WStatCountsStatistic(counts_on, counts_off, alpha, mu_signal)
            self.instrument_spectral_info["TS_H0"] += np.nansum(stat.stat_null.ravel())

        if len(dl4_dl5_steps) > 0:
            self.log.info("Perform DL4 to DL5 processes!")

            for step in dl4_dl5_steps:
                analysis_step = AnalysisStep.create(step, self.config, **kwargs)

                analysis_step.run(
                    datasets=self.datasets, instrument_spectral_info=self.instrument_spectral_info
                )

                # Update the final data product objects
                for data_product in self.final_data_products:
                    if hasattr(analysis_step, data_product):
                        setattr(self, data_product, getattr(analysis_step, data_product))

        if self.fit_result:
            self.instrument_spectral_info, message = get_goodness_of_fit_stats(
                self.instrument_spectral_info
            )
            self.log.info(message)

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

    def update_config(self, config):
        """Update the primary config with another config."""
        self.config = self.config.update(config=config)
