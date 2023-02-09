"""
Config-driven high level analysis interface.
"""
import logging

from gammapy.datasets import Datasets
from gammapy.modeling.models import Models

from asgardpy.config.generator import AsgardpyConfig
from asgardpy.data.base import AnalysisStep
from asgardpy.data.target import set_models

# import numpy as np


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
        self.spectral_energy_ranges = []
        self.dataset_name_list = []

        self.final_model = []
        self.final_data_products = ["fit", "fit_result", "flux_points", "light_curve"]

        for data_product in self.final_data_products:
            setattr(self, data_product, None)

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
            # Should consider a better approach in running the steps in an order.
            if "datasets" in step:
                analysis_step = AnalysisStep.create(step, self.config, **kwargs)
                datasets_list, models_list, energy_edges = analysis_step.run()

                if step == "datasets-3d":
                    # Get the final model only from a complete 3D dataset, to use it for other 1D datasets.

                    ## New: Read each models list, to get all datasets_names, and
                    ## then in a separate step, unify the models to be attached to the final joint datasets object
                    # for data in datasets_list:
                    # Make a check to see if all component types of SkyModels
                    # are present throughout all datasets

                    # if data.models is not None
                    # for models in models_list:
                    models_list = Models(models_list)
                    # print(models_list, "is the model list")
                    target_source_model = models_list[self.config.target.source_name]
                    print(
                        "Dataset names in the target model are:", target_source_model.datasets_names
                    )

                    # for m in mask:
                    #    self.ex_masks.append(m)
                    if target_source_model.spatial_model:  ## Re-evaluate
                        # If the target source has Spatial model included, only then (?) get all the models as final_model
                        for m in models_list:
                            # print(m.name)
                            self.final_model.append(m)
                    else:
                        print("The target source only has spectral model:", target_source_model)

                    if target_source_model.datasets_names:  # Models object and not DatasetModels
                        # To get all datasets_names from the target source model only.
                        for names in target_source_model.datasets_names:
                            print("Included for the target source is dataset of name:", names)
                            if names not in self.dataset_name_list:
                                # Only update the list if a new name is found.
                                self.dataset_name_list.append(names)
                    else:
                        print(
                            "Check if there are no other way dataset names are stored in target model :",
                            target_source_model,
                        )
                    # Finally, simply update the final datasets list
                    for data in datasets_list:
                        self.datasets.append(data)

                if step == "datasets-1d":
                    for data in datasets_list:
                        # if data.models:
                        # target_source_model = data.models[self.config.target.source_name]
                        # print("Dataset names in the target model are:", target_source_model.datasets_names)
                        # for names in target_source_model.names, ie, when it is not an Models object
                        # This is for 1D Datasets for now.
                        # print("Checking the models object in the data of type", type(data), " has models attribute as", data.models, "with type", type(data.models))
                        print("Before enlisting from a 1D dataset", self.dataset_name_list)
                        if data.name not in self.dataset_name_list:
                            self.dataset_name_list.append(data.name)
                        print("After enlisting from a 1D dataset", self.dataset_name_list)
                        self.datasets.append(data)

                        # self.dataset_name_list.append(target_source_model.names)

                        # else:
                        # Looking up models from the config info, and assigning it to these datasets.
                        #    data = set_models(self.config.target, data)
                        """
                        if self.final_model is not None:
                            #print("Assining the (full?) final model to a dataset without any spatial model...")
                            print(data.models, type(data.models))
                            print(data.models[0].spatial_model)
                            # Assinging only spatial model.
                            ## data.models[0].spatial_model = self.final_model[self.config.target.source_name].spatial_model
                            data.models[0].spectral_model = self.final_model[self.config.target.source_name].spectral_model
                            #data = set_models(self.config.target, data, self.final_model)
                            print("Checking if assigned or not...", data.models, type(data.models), self.final_model[self.config.target.source_name])
                            print(data)
                        """
                        # Finally, simply update the final datasets list
                        # for data in datasets_list:

                        # self.dataset_name_list = np.unique(np.array(self.dataset_name_list))
                        # else:
                        # For 1D Datasets

                        # data = set_models(
                        #    config=self.config.target, datasets=self.datasets, models=self.final_model
                        # )

                        # if self.final_model is not None:
                        # Setting the datasets name for the target source.
                        # if not models_list:
                        #    print(models_list, "1D dataset gives some models")
                        # self.final_model.append(m for m in models_list)

                for edges in energy_edges:
                    # Update the spectral energy ranges for each dataset.
                    self.spectral_energy_ranges.append(edges)
            else:
                # Running DL4 functions on a given Datasets object.
                if step == "fit":
                    # Confirming the Final Models object for all the datasets
                    # for the Fit function.
                    self.final_model = Models(self.final_model)
                    self.log.info(f"Full final models list is {self.final_model}")
                    # self.final_model = Models(self.final_model[0])
                    # self.log.info(f"Full final models list is {self.final_model}")
                    # self.log.info(f"The final model for target source, used is {self.final_model[self.config.target.source_name]}")
                    # self.log.info(f"The full list of dataset names is {self.dataset_name_list}")
                    # self.log.info(f"The full dataset is:")
                    # for data in self.datasets:
                    #    print(data)
                    self.final_model[
                        self.config.target.source_name
                    ].datasets_names = self.dataset_name_list
                    self.log.info(
                        f"The final model for target source, used is {self.final_model[self.config.target.source_name]}"
                    )
                    for data in self.datasets:
                        print("Before model assignment", data)

                        data.models = self.final_model
                        """
                        for m in self.final_model:
                            if "diffuse" not in m.name:
                                sig = data.npred_signal(model_name=m.name).data.sum()
                                print("For model of", m.name, "predicted signal is", sig)
                            #if m.name == self.config.target.source_name:
                            #    data.npred_signal() = data.npred_signal(model_name=m.name)
                            #    print("After trying to assign npred_signal value, the total npred of the dataset is:", data.npred().data.sum())
                        #data.npred()
                        #print(data.npred_signal(model_name=self.config.target.source_name).data.sum())
                        print(data.npred().data.sum())
                        #data.mask_safe = mask_safe
                        """
                        print("After model assignment", data)
                    self.log.info(f"After models assignment, the full dataset is {self.datasets}")
                    self.log.info(
                        "Final model for target source is updated with the full datasets list"
                    )

                    # for data in self.datasets:
                    #    print(data)
                    # self.datasets = set_models(
                    #    config=self.config.target, datasets=self.datasets, models=self.final_model
                    # )
                    # print(self.datasets)
                    """
                    for data in self.datasets:
                        #print(data)
                        print(data.models.names)
                        if data.models.names == [self.config.target.source_name]:
                            print("The dataset has model of the target source with datasets names, ", data.models[0].datasets_names)
                            #data.models[0].datasets_names = self.dataset_name_list
                            print("The dataset has model of the target source with datasets names, now, ", data.models[0].datasets_names)
                        #else:
                            #data.models[self.config.target.source_name].datasets_names = self.dataset_name_list
                    #    data.models[self.config.target.source_name] = self.final_model[self.config.target.source_name]
                    for data in self.datasets:
                        print("Final models are for dataset", data.name, data.models)
                    #for m in self.datasets.models:
                    #    print(m.datasets_names)
                    """
                analysis_step = AnalysisStep.create(step, self.config, **kwargs)
                analysis_step.run(datasets=self.datasets, energy_ranges=self.spectral_energy_ranges)

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
