"""
Config-driven high level analysis interface.
"""
import logging

from gammapy.datasets import Datasets
from gammapy.modeling.models import Models

from asgardpy.base import AnalysisStep
from asgardpy.config import AsgardpyConfig
from asgardpy.data import apply_selection_mask_to_models, set_models

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
        self.instrument_spectral_info = {"name": [], "spectral_energy_ranges": []}
        self.dataset_name_list = []

        self.final_model = Models()
        self.final_data_products = ["fit", "fit_result", "flux_points"]

        for data_product in self.final_data_products:
            setattr(self, data_product, None)

    @property
    def models(self):
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
        if steps is None:
            steps = self.config.general.steps
            overwrite = self.config.general.overwrite
        else:
            if overwrite is None:
                overwrite = True
        for step in steps:
            if "datasets" in step:
                analysis_step = AnalysisStep.create(step, self.config, **kwargs)
                datasets_list, models_list, instrument_spectral_info = analysis_step.run()

                if step == "datasets-3d":
                    target_source_model = models_list[self.config.target.source_name]

                    if target_source_model.spatial_model:
                        # If the target source has Spatial model included,
                        # only then (?) get all the models as final_model.
                        # Needs reconsideration.
                        for m in models_list:
                            self.final_model.append(m)
                    else:
                        self.log.info(
                            f"The target source only has spectral model:{target_source_model}"
                        )

                    # To get all datasets_names from the target source model.
                    if target_source_model.datasets_names:
                        for names in target_source_model.datasets_names:
                            # Only update the list if a new name is found.
                            if names not in self.dataset_name_list:
                                self.dataset_name_list.append(names)

                    # Finally, simply update the final datasets list
                    for data in datasets_list:
                        self.datasets.append(data)

                if step == "datasets-1d":
                    for data in datasets_list:
                        if data.name not in self.dataset_name_list:
                            self.dataset_name_list.append(data.name)
                        self.datasets.append(data)

                # Update the name and spectral energy ranges for each
                # instrument Datasets, to be used for the FluxPointsAnalysisStep.
                for name in instrument_spectral_info["name"]:
                    self.instrument_spectral_info["name"].append(name)

                for edges in instrument_spectral_info["spectral_energy_ranges"]:
                    self.instrument_spectral_info["spectral_energy_ranges"].append(edges)

            else:
                # Running DL5 functions on a given Datasets object.
                if step == "fit":
                    # Confirming the Final Models object for all the datasets
                    # for the Fit function.

                    # In case of only 1D dataset being selected, Model is read
                    # from the config information, by passing None as value.
                    # It should be reinitialized again with the DatasetModels.
                    if len(self.final_model) == 0:
                        self.final_model = None
                    # Apply selection filter for the ROI if more than 1 models
                    # are provided in the FoV, that are not the background models.
                    elif len(self.final_model) > 1:
                        self.final_model = apply_selection_mask_to_models(
                            list_sources=self.final_model,
                            target_source=self.config.target.source_name,
                            roi_radius=self.config.target.roi_selection.roi_radius,
                            free_sources=self.config.target.roi_selection.free_sources,
                        )

                    self.datasets = set_models(
                        self.config.target,
                        self.datasets,
                        self.dataset_name_list,
                        models=self.final_model,
                        target_source_name=self.config.target.source_name,
                    )
                    if self.final_model is None:
                        self.final_model = self.datasets.models

                analysis_step = AnalysisStep.create(step, self.config, **kwargs)
                analysis_step.run(
                    datasets=self.datasets, instrument_spectral_info=self.instrument_spectral_info
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

    def update_config(self, config):
        self.config = self.config.update(config=config)
