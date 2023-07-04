"""
Main classes to define 1D Dataset Config, 1D Dataset Analysis Step and
to generate 1D Datasets from given Instruments' DL3 data from the config.
"""

import logging
from typing import List

import numpy as np
from astropy import units as u

# from gammapy.analysis import Analysis, AnalysisConfig - no support for DL3 with RAD_MAX
from gammapy.datasets import Datasets
from gammapy.makers import SpectrumDatasetMaker

from asgardpy.base import (
    AnalysisStepBase,
    BackgroundConfig,
    BaseConfig,
    GeomConfig,
    MapAxesConfig,
    MapSelectionEnum,
    ObservationsConfig,
    ReductionTypeEnum,
    SafeMaskConfig,
    SkyPositionConfig,
    get_energy_axis,
)
from asgardpy.base.geom import generate_geom, get_source_position
from asgardpy.base.reduction import (
    generate_dl4_dataset,
    get_bkg_maker,
    get_dataset_template,
    get_filtered_observations,
    get_safe_mask_maker,
)
from asgardpy.io import DL3Files, InputConfig

__all__ = [
    "Datasets1DAnalysisStep",
    "Dataset1DBaseConfig",
    "Dataset1DConfig",
    "Dataset1DGeneration",
    "Dataset1DInfoConfig",
]

log = logging.getLogger(__name__)


# Defining various components of 1D Dataset Config section
class Dataset1DInfoConfig(BaseConfig):
    name: str = "dataset-name"
    geom: GeomConfig = GeomConfig()
    observation: ObservationsConfig = ObservationsConfig()
    background: BackgroundConfig = BackgroundConfig()
    safe_mask: SafeMaskConfig = SafeMaskConfig()
    on_region: SkyPositionConfig = SkyPositionConfig()
    containment_correction: bool = True
    map_selection: List[MapSelectionEnum] = []
    spectral_energy_range: MapAxesConfig = MapAxesConfig()


class Dataset1DBaseConfig(BaseConfig):
    name: str = "Instrument-name"
    io: List[InputConfig] = [InputConfig()]
    dataset_info: Dataset1DInfoConfig = Dataset1DInfoConfig()


class Dataset1DConfig(BaseConfig):
    type: ReductionTypeEnum = ReductionTypeEnum.spectrum
    instruments: List[Dataset1DBaseConfig] = [Dataset1DBaseConfig()]


# The main Analysis Step
class Datasets1DAnalysisStep(AnalysisStepBase):
    """
    From the given config information, prepare the full list of 1D datasets,
    iterating over all the Instruments' information by running the
    Dataset1DGeneration function.
    """

    tag = "datasets-1d"

    def _run(self):
        instruments_list = self.config.dataset1d.instruments
        self.log.info(f"{len(instruments_list)} number of 1D Datasets given")

        datasets_1d_final = Datasets()
        instrument_spectral_info = {"name": [], "spectral_energy_ranges": []}

        # Iterate over all instrument information given:
        for i in np.arange(len(instruments_list)):
            config_1d_dataset = instruments_list[i]
            instrument_spectral_info["name"].append(config_1d_dataset.name)

            generate_1d_dataset = Dataset1DGeneration(self.log, config_1d_dataset, self.config)
            dataset = generate_1d_dataset.run()

            # Get the spectral energy information for each Instrument Dataset
            energy_axes = config_1d_dataset.dataset_info.spectral_energy_range
            if len(energy_axes.axis_custom.edges) > 0:
                energy_bin_edges = get_energy_axis(energy_axes, only_edges=True, custom_range=True)
            else:
                energy_bin_edges = get_energy_axis(
                    energy_axes,
                    only_edges=True,
                )
            instrument_spectral_info["spectral_energy_ranges"].append(energy_bin_edges)

            if self.config.general.stacked_dataset:
                dataset = dataset.stack_reduce(name=config_1d_dataset.name)
                datasets_1d_final.append(dataset)
            else:
                for data in dataset:
                    datasets_1d_final.append(data)

        return (
            datasets_1d_final,
            None,
            instrument_spectral_info,
        )


class Dataset1DGeneration:
    """
    Class for 1D dataset creation based on the config or AsgardpyConfig
    information provided on the 1D dataset and the target source.

    Runs the following steps:

    1. Read the DL3 files of 1D datasets into DataStore object.

    2. Perform any Observation selection, based on Observation IDs or time intervals.

    3. Create the base dataset template, including the main counts geometry.

    4. Prepare standard data reduction makers using the parameters passed in the config.

    5. Generate the final dataset.
    """

    def __init__(self, log, config_1d_dataset, config_full):
        self.config_1d_dataset_io = config_1d_dataset.io
        self.log = log
        self.config_1d_dataset_info = config_1d_dataset.dataset_info
        self.config_target = config_full.target
        self.n_jobs = config_full.general.n_jobs
        self.parallel_backend = config_full.general.parallel_backend
        self.exclusion_regions = []
        self.datasets = Datasets()

    def run(self):
        # First check for the given file list if they are readable or not.
        file_list = {}
        dl3_info = DL3Files(
            self.config_1d_dataset_io[0],
            file_list,
            log=self.log,
        )
        dl3_info.list_dl3_files()

        # Applying all provided filters to get the Observations object
        observations = get_filtered_observations(
            dl3_path=self.config_1d_dataset_io[0].input_dir,
            obs_config=self.config_1d_dataset_info.observation,
            log=self.log,
        )
        # Get dict information of the ON region, with its SkyCoord position and angular radius
        center_pos = get_source_position(config_target=self.config_target)

        # Create the main counts geometry
        geom = generate_geom(
            tag="1d", geom_config=self.config_1d_dataset_info.geom, center_pos=center_pos
        )
        dataset_template = get_dataset_template(
            tag="1d", geom=geom, geom_config=self.config_1d_dataset_info.geom
        )

        # Get all the Dataset reduction makers
        dataset_maker = SpectrumDatasetMaker(
            containment_correction=self.config_1d_dataset_info.containment_correction,
            selection=self.config_1d_dataset_info.map_selection,
        )
        bkg_maker = get_bkg_maker(
            bkg_config=self.config_1d_dataset_info.background,
            geom_config=self.config_1d_dataset_info.geom,
            exclusion_regions=self.exclusion_regions,
            config_target=self.config_target,
            log=self.log,
        )
        safe_maker = get_safe_mask_maker(safe_config=self.config_1d_dataset_info.safe_mask)

        # Produce the final Dataset
        self.datasets = generate_dl4_dataset(
            tag="1d",
            observations=observations,
            dataset_template=dataset_template,
            dataset_maker=dataset_maker,
            bkg_maker=bkg_maker,
            safe_maker=safe_maker,
            n_jobs=self.n_jobs,
            parallel_backend=self.parallel_backend,
        )
        self.update_dataset(observations, safe_maker)

        return self.datasets

    def update_dataset(self, observations, safe_maker):
        """ """
        safe_cfg = self.config_1d_dataset_info.safe_mask
        pars = safe_cfg.parameters

        for data, obs in zip(self.datasets, observations):
            # Rename the datasets using the appropriate Obs ID
            data._name = obs.obs_id

            # Use custom safe energy mask
            if safe_maker is None:
                data.mask_safe = data.counts.geom.energy_mask(
                    energy_min=u.Quantity(pars["min"]),
                    energy_max=u.Quantity(pars["max"]),
                )
