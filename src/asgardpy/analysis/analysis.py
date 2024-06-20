"""
Config-driven high level analysis interface.
"""

import logging

from gammapy.datasets import Datasets
from gammapy.modeling.models import Models, SkyModel
from pydantic import ValidationError

from asgardpy.analysis.step import AnalysisStep
from asgardpy.config.generator import AsgardpyConfig, gammapy_to_asgardpy_model_config
from asgardpy.data.target import set_models
from asgardpy.stats.stats import get_goodness_of_fit_stats

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
            try:
                other_config = AsgardpyConfig.read(self.config.target.models_file)
                self.config = self.config.update(other_config)
            except ValidationError:
                self.config = gammapy_to_asgardpy_model_config(
                    gammapy_model=self.config.target.models_file,
                    asgardpy_config_file=self.config,
                )

        self.config.set_logging()
        self.datasets = Datasets()
        self.instrument_spectral_info = {
            "name": [],
            "spectral_energy_ranges": [],
            "en_bins": 0,
            "free_params": 0,
            "DoF": 0,
        }
        self.dataset_name_list = []

        self.final_model = Models()
        self.final_data_products = ["fit", "fit_result", "flux_points", "model_deabs", "flux_points_deabs"]

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

    def update_models_list(self, models_list):
        """
        This function updates the final_model object with a given list of
        models, only if the target source has a Spatial model included.

        This step is only valid for 3D Datasets, and might need reconsideration
        in the future.
        """
        if models_list:
            target_source_model = models_list[self.config.target.source_name]

            if target_source_model.spatial_model:
                for model_ in models_list:
                    self.final_model.append(model_)
            else:  # pragma: no cover
                self.log.info(
                    "The target source %s only has spectral model",
                    self.config.target.source_name,
                )

    def get_correct_intrinsic_model(self):
        """
        After running the Fit analysis step, one can retrieve the correct
        covariance matrix for only the intrinsic or EBL-deabsorbed spectra.

        This function will be redundant in Gammapy v1.3
        """
        temp_model = self.final_model[0].spectral_model.model1

        cov_matrix = self.fit.covariance(
            datasets=self.datasets, optimize_result=self.fit_result.optimize_result
        ).matrix

        temp_model.covariance = cov_matrix[: len(temp_model.parameters), : len(temp_model.parameters)]

        self.model_deabs = SkyModel(
            spectral_model=temp_model,
            name=self.final_model[0].name,
            datasets_names=self.final_model[0].datasets_names,
        )

    def get_correct_ebl_deabs_flux_points(self):
        """
        After running the get_correct_intrinsic_model function, this function
        will re-run the flux-points analysis step again, by using only the
        intrinsic model as the reference spectral model.

        This function will be redundant in Gammapy v1.3
        """
        temp_fp = self.flux_points
        self.flux_points = None

        if not self.model_deabs:
            self.get_correct_intrinsic_model()

        self.run(["flux-points"])

        for fp in self.flux_points:
            fp._models = self.model_deabs
            fp._reference_model = self.model_deabs

        self.flux_points_deabs = self.flux_points
        self.flux_points = temp_fp

    def add_to_instrument_info(self, info_dict):
        """
        Update the name, Degrees of Freedom and spectral energy ranges for each
        instrument Datasets, to be used for the DL4 to DL5 processes.
        """
        for name in info_dict["name"]:
            self.instrument_spectral_info["name"].append(name)

        for edges in info_dict["spectral_energy_ranges"]:
            self.instrument_spectral_info["spectral_energy_ranges"].append(edges)

        self.instrument_spectral_info["en_bins"] += info_dict["en_bins"]
        self.instrument_spectral_info["free_params"] += info_dict["free_params"]

    def update_dof_value(self):
        """
        Simple function to update total Degrees of Freedom value in the
        instrument_spectral_info dict from a filled final_model object.
        """
        if len(self.final_model) > 0:
            # Add to the total number of free model parameters
            n_free_params = len(list(self.final_model.parameters.free_parameters))
            self.instrument_spectral_info["free_params"] += n_free_params

            # Get the final degrees of freedom as en_bins - free_params
            self.instrument_spectral_info["DoF"] = (
                self.instrument_spectral_info["en_bins"] - self.instrument_spectral_info["free_params"]
            )

    def run(self, steps=None, **kwargs):
        """
        Main function to run the AnalaysisSteps provided.

        Currently overwrite option is used from the value in AsgardpyConfig,
        and is True by default.
        """
        if steps is None:
            steps = self.config.general.steps
            self.overwrite = self.config.general.overwrite

        dl3_dl4_steps = [step for step in steps if "datasets" in step]
        dl4_dl5_steps = [step for step in steps if "datasets" not in step]

        if len(dl3_dl4_steps) > 0:
            self.log.info("Perform DL3 to DL4 process!")

            for step in dl3_dl4_steps:
                analysis_step = AnalysisStep.create(step, self.config, **kwargs)

                datasets_list, models_list, instrument_spectral_info = analysis_step.run()

                self.update_models_list(models_list)

                # To get all datasets_names from the datasets and update the final datasets list
                for data in datasets_list:
                    if data.name not in self.dataset_name_list:
                        self.dataset_name_list.append(data.name)
                    self.datasets.append(data)

                self.add_to_instrument_info(instrument_spectral_info)

            self.datasets, self.final_model = set_models(
                self.config.target,
                self.datasets,
                self.dataset_name_list,
                models=self.final_model,
            )
            self.log.info("Models have been associated with the Datasets")

            self.update_dof_value()

        if len(dl4_dl5_steps) > 0:
            self.log.info("Perform DL4 to DL5 processes!")

            for step in dl4_dl5_steps:
                analysis_step = AnalysisStep.create(step, self.config, **kwargs)

                analysis_step.run(datasets=self.datasets, instrument_spectral_info=self.instrument_spectral_info)

                # Update the final data product objects
                for data_product in self.final_data_products:
                    if hasattr(analysis_step, data_product):
                        setattr(self, data_product, getattr(analysis_step, data_product))

        if self.fit_result:
            self.instrument_spectral_info, message = get_goodness_of_fit_stats(
                self.datasets, self.instrument_spectral_info
            )
            self.log.info(message)

    # keep these methods to be backward compatible
    def get_1d_datasets(self):
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
