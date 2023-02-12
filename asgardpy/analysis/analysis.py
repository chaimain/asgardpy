"""
Config-driven high level analysis interface.
"""
import logging

from gammapy.datasets import Datasets
from gammapy.modeling.models import Models

from asgardpy.config.generator import AsgardpyConfig
from asgardpy.data.base import AnalysisStep
from asgardpy.data.target import set_models


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
        self.instrument_spectral_info = {"name": [], "spectral_energy_ranges": []}
        self.dataset_name_list = []

        self.final_model = Models()
        self.final_data_products = ["fit", "fit_result", "flux_points", "light_curve"]

        for data_product in self.final_data_products:
            setattr(self, data_product, None)

    @property
    def models(self):
        if not self.datasets:
            raise RuntimeError("No datasets defined. Impossible to set models.")
        return self.datasets.models

    """
    @models.setter
    def models(self, models):

        # Set a given Model to the final datasets object.

        self.datasets = set_models(
            config=self.config.target,
            datasets=self.datasets,
            models=models,
            extend=False
        )
    """

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
            # Should consider a better approach in running the steps in an order.
            if "datasets" in step:
                analysis_step = AnalysisStep.create(step, self.config, **kwargs)
                datasets_list, models_list, instrument_spectral_info = analysis_step.run()

                if step == "datasets-3d":
                    # Get the final model only from a complete 3D dataset,
                    # to use it for other 1D datasets.

                    # New: Read each models list, to get all datasets_names, and
                    # then in a separate step, unify the models to be attached
                    # to the final joint datasets object

                    # Make a check to see if all component types of SkyModels
                    # are present throughout all datasets

                    target_source_model = models_list[self.config.target.source_name]

                    if target_source_model.spatial_model:  # Re-evaluate
                        # If the target source has Spatial model included,
                        # only then (?) get all the models as final_model
                        for m in models_list:
                            self.final_model.append(m)
                    else:
                        self.log.info(f"The target source only has spectral model:{target_source_model}")

                    if target_source_model.datasets_names:
                        # To get all datasets_names from the target source model only.
                        for names in target_source_model.datasets_names:
                            if names not in self.dataset_name_list:
                                # Only update the list if a new name is found.
                                self.dataset_name_list.append(names)
                    else:
                        self.log.info(
                            "Check if there are no other way, the dataset names"
                            f"are stored in target model :{target_source_model}"
                        )
                    # Finally, simply update the final datasets list
                    for data in datasets_list:
                        self.datasets.append(data)

                if step == "datasets-1d":
                    for data in datasets_list:
                        if data.name not in self.dataset_name_list:
                            self.dataset_name_list.append(data.name)
                        self.log.info(f"The list of names of selected datasets are: {self.dataset_name_list}")
                        self.datasets.append(data)

                # Update the name and spectral energy ranges for each
                # instrument Datasets, to be used for the FluxPointsAnalysisStep.
                for name in instrument_spectral_info["name"]:
                    self.instrument_spectral_info["name"].append(name)
                for edges in instrument_spectral_info["spectral_energy_ranges"]:
                    self.instrument_spectral_info["spectral_energy_ranges"].append(edges)

            else:
                # Running DL4 functions on a given Datasets object.
                if step == "fit":
                    # Confirming the Final Models object for all the datasets
                    # for the Fit function.
                    self.log.info(f"Full final models list is {self.final_model}")
                    self.datasets = set_models(
                        self.config.target, self.datasets, self.dataset_name_list,
                        models=self.final_model,
                        target_source_name=self.config.target.source_name,
                    )
                    self.log.info(
                        f"After models assignment, the full dataset is {self.datasets}"
                    )

                analysis_step = AnalysisStep.create(step, self.config, **kwargs)
                analysis_step.run(
                    datasets=self.datasets,
                    instrument_spectral_info=self.instrument_spectral_info
                )

                # Update the final data product objects
                for data_product in self.final_data_products:
                    if hasattr(analysis_step, data_product):
                        setattr(self, data_product, getattr(analysis_step, data_product))

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
