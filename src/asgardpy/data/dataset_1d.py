"""
Main classes to define 1D Dataset Config, 1D Dataset Analysis Step and
to generate 1D Datasets from given Instruments' DL3 data from the config.
"""

import logging

import numpy as np
from astropy import units as u
from gammapy.datasets import Datasets

from asgardpy.analysis.step_base import AnalysisStepBase
from asgardpy.base.base import BaseConfig
from asgardpy.base.geom import (
    GeomConfig,
    SkyPositionConfig,
    generate_geom,
    get_source_position,
)
from asgardpy.base.reduction import (
    BackgroundConfig,
    MapSelectionEnum,
    ObservationsConfig,
    ReductionTypeEnum,
    SafeMaskConfig,
    generate_dl4_dataset,
    get_bkg_maker,
    get_dataset_maker,
    get_dataset_reference,
    get_exclusion_region_mask,
    get_filtered_observations,
    get_safe_mask_maker,
)
from asgardpy.io.input_dl3 import InputDL3Config  # DL3Files
from asgardpy.io.io_dl4 import DL4BaseConfig, DL4Files, get_reco_energy_bins
from asgardpy.version import __public_version__

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
    """Config section for 1D DL3 Dataset Reduction for each instrument."""

    name: str = "dataset-name"
    geom: GeomConfig = GeomConfig()
    observation: ObservationsConfig = ObservationsConfig()
    background: BackgroundConfig = BackgroundConfig()
    safe_mask: SafeMaskConfig = SafeMaskConfig()
    on_region: SkyPositionConfig = SkyPositionConfig()
    containment_correction: bool = True
    map_selection: list[MapSelectionEnum] = []


class Dataset1DBaseConfig(BaseConfig):
    """
    Config section for 1D DL3 Dataset base information for each instrument.
    """

    name: str = "Instrument-name"
    input_dl3: list[InputDL3Config] = [InputDL3Config()]
    input_dl4: bool = False
    dataset_info: Dataset1DInfoConfig = Dataset1DInfoConfig()
    dl4_dataset_info: DL4BaseConfig = DL4BaseConfig()


class Dataset1DConfig(BaseConfig):
    """Config section for a list of all 1D DL3 Datasets information."""

    type: ReductionTypeEnum = ReductionTypeEnum.spectrum
    instruments: list[Dataset1DBaseConfig] = [Dataset1DBaseConfig()]


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
        self.log.info("%d number of 1D Datasets given", len(instruments_list))

        datasets_1d_final = Datasets()
        instrument_spectral_info = {"name": [], "spectral_energy_ranges": []}

        # Calculate the total number of reconstructed energy bins used
        en_bins = 0

        # Iterate over all instrument information given:
        for i in np.arange(len(instruments_list)):
            config_1d_dataset = instruments_list[i]
            instrument_spectral_info["name"].append(config_1d_dataset.name)
            dl4_files = DL4Files(config_1d_dataset.dl4_dataset_info, self.log)

            if not config_1d_dataset.input_dl4:
                generate_1d_dataset = Dataset1DGeneration(self.log, config_1d_dataset, self.config)
                dataset = generate_1d_dataset.run()
            else:
                dataset = dl4_files.get_dl4_dataset(config_1d_dataset.dataset_info.observation)

            energy_bin_edges = dl4_files.get_spectral_energies()
            instrument_spectral_info["spectral_energy_ranges"].append(energy_bin_edges)

            if self.config.general.stacked_dataset:
                dataset = dataset.stack_reduce(name=config_1d_dataset.name)
                dataset._meta.optional = {
                    "instrument": config_1d_dataset.name,
                }
                dataset._meta.creation.creator += f", Asgardpy {__public_version__}"

                en_bins = get_reco_energy_bins(dataset, en_bins)
                datasets_1d_final.append(dataset)
            else:
                for data in dataset:
                    data._meta.optional = {
                        "instrument": config_1d_dataset.name,
                    }
                    data._meta.creation.creator += f", Asgardpy {__public_version__}"
                    en_bins = get_reco_energy_bins(data, en_bins)
                    datasets_1d_final.append(data)

        instrument_spectral_info["en_bins"] = en_bins

        # No linked model parameters or other free model parameters taken here
        instrument_spectral_info["free_params"] = 0

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

    3. Create the base dataset reference, including the main counts geometry.

    4. Prepare standard data reduction makers using the parameters passed in the config.

    5. Generate the final dataset.
    """

    def __init__(self, log, config_1d_dataset, config_full):
        self.config_1d_dataset_io = config_1d_dataset.input_dl3
        self.log = log
        self.config_1d_dataset_info = config_1d_dataset.dataset_info
        self.config_target = config_full.target
        self.n_jobs = config_full.general.n_jobs
        self.parallel_backend = config_full.general.parallel_backend
        self.exclusion_regions = []
        self.datasets = Datasets()

    def run(self):
        """
        Main function to run the creation of 1D dataset.
        """
        # First check for the given file list if they are readable or not.
        # dl3_info = DL3Files(
        #     self.config_1d_dataset_io[0],
        #     log=self.log,
        # )
        # dl3_info.list_dl3_files()

        # if len(dl3_info.events_files) == 0:
        #     self.log.error("No DL3 files found at %s", dl3_info.dl3_path)

        # Applying all provided filters to get the Observations object
        observations = get_filtered_observations(
            dl3_path=self.config_1d_dataset_io[0].input_dir,
            obs_config=self.config_1d_dataset_info.observation,
            log=self.log,
        )
        # Get dict information of the ON region, with its SkyCoord position and angular radius
        center_pos = get_source_position(target_region=self.config_1d_dataset_info.on_region)

        # Create the main counts geometry
        geom = generate_geom(tag="1d", geom_config=self.config_1d_dataset_info.geom, center_pos=center_pos)

        # Get all the Dataset reduction makers
        dataset_reference = get_dataset_reference(
            tag="1d", geom=geom, geom_config=self.config_1d_dataset_info.geom
        )

        dataset_maker = get_dataset_maker(
            tag="1d",
            dataset_config=self.config_1d_dataset_info,
        )

        safe_maker = get_safe_mask_maker(safe_config=self.config_1d_dataset_info.safe_mask)

        excluded_geom = generate_geom(
            tag="1d-ex", geom_config=self.config_1d_dataset_info.geom, center_pos=center_pos
        )
        exclusion_mask = get_exclusion_region_mask(
            exclusion_params=self.config_1d_dataset_info.background.exclusion,
            exclusion_regions=self.exclusion_regions,
            excluded_geom=excluded_geom,
            config_target=self.config_target,
            geom_config=self.config_1d_dataset_info.geom,
            log=self.log,
        )

        bkg_maker = get_bkg_maker(
            bkg_config=self.config_1d_dataset_info.background,
            exclusion_mask=exclusion_mask,
        )

        # Produce the final Dataset
        self.datasets = generate_dl4_dataset(
            tag="1d",
            observations=observations,
            dataset_reference=dataset_reference,
            dataset_maker=dataset_maker,
            bkg_maker=bkg_maker,
            safe_maker=safe_maker,
            n_jobs=self.n_jobs,
            parallel_backend=self.parallel_backend,
        )
        self.update_dataset(observations)

        return self.datasets

    def update_dataset(self, observations):
        """
        Update the datasets generated by DatasetsMaker with names as per the
        Observation ID and if a custom safe energy mask is provided in the
        config, apply it to each dataset accordingly.
        """
        safe_cfg = self.config_1d_dataset_info.safe_mask
        pars = safe_cfg.parameters

        for data, obs in zip(self.datasets, observations, strict=True):
            # Rename the datasets using the appropriate Obs ID
            data._name = str(obs.obs_id)

            # Use custom safe energy mask
            if "custom-mask" in safe_cfg.methods:
                data.mask_safe = data.counts.geom.energy_mask(
                    energy_min=u.Quantity(pars["min"]), energy_max=u.Quantity(pars["max"]), round_to_edge=True
                )
