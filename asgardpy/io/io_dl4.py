"""
Basic classes defining Input Config for DL4 files and some functions to
retrieve information for the DL4 to DL5 processes.
"""

import logging
import re
from enum import Enum
from pathlib import Path

from gammapy.datasets import DATASET_REGISTRY, Datasets

from asgardpy.base.base import BaseConfig, PathType
from asgardpy.base.geom import MapAxesConfig, get_energy_axis

__all__ = [
    "InputDL4Config",
    "DL4Files",
    "DL4BaseConfig",
    "get_reco_energy_bins",
]


class DatasetTypeEnum(str, Enum):
    """
    Config section for list of Dataset types in Gammapy.
    """

    MapDataset = "MapDataset"
    MapDatasetOnOff = "MapDatasetOnOff"
    SpectrumDataset = "SpectrumDataset"
    SpectrumDatasetOnOff = "SpectrumDatasetOnOff"
    FluxPointsDataset = "FluxPointsDataset"


class DL4FormatEnum(str, Enum):
    """
    Config section for list of formats for Datasets in Gammapy.
    """

    ogip = "ogip"
    ogip_sherpa = "ogip-sherpa"
    gadf = "gadf"
    gadf_sed = "gadf-sed"


class InputDL4Config(BaseConfig):
    """
    Config section for main information on getting the relevant DL4 files.
    """

    type: DatasetTypeEnum = DatasetTypeEnum.MapDataset
    input_dir: PathType = PathType("None")
    # Can be OGIP format (Stacked or unstacked obs) or fits format (stacked obs)
    glob_pattern: str = "pha*fits"
    dl4_format: DL4FormatEnum = DL4FormatEnum.gadf


class DL4BaseConfig(BaseConfig):
    """Config section for DL4 Dataset for a given instrument."""

    dl4_dataset: InputDL4Config = InputDL4Config()
    spectral_energy_range: MapAxesConfig = MapAxesConfig()


# Main class for DL4 I/O
class DL4Files:
    """
    A general class to retrieve information from given DL4 files.
    """

    def __init__(self, dl4_dataset_info, log=None):
        self.dl4_dataset_info = dl4_dataset_info
        self.dl4_dataset = dl4_dataset_info.dl4_dataset
        self.dl4_type = self.dl4_dataset.type
        self.dl4_path = None
        self.dl4_file = None

        self._check_dl4_type()

        if Path(self.dl4_dataset.input_dir).exists():
            if Path(self.dl4_dataset.input_dir).is_file():
                self.dl4_file = Path(self.dl4_dataset.input_dir)
            else:
                self.dl4_path = Path(self.dl4_dataset.input_dir)
        else:
            self.log.error("%s is not a valid file location", self.dl4_dataset.input_dir)

        if not log:
            self._set_logging()
        else:
            self.log = log

    def _set_logging(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

    def _check_dl4_type(self):
        try:
            DATASET_REGISTRY.get_cls(self.dl4_type)
        except KeyError:
            self.log.error("Incorrect DL4 Type passed!")

    def get_dl4_files(self, observation_config):
        """
        Fetch the required DL4 files from the given directory path, file glob
        search and possible list of observation ids to select the dataset files
        from the full list in th directory.
        """
        dl4_file_list = []
        all_dl4_files = sorted(list(self.dl4_path.glob(self.dl4_dataset.glob_pattern)))

        if len(all_dl4_files) == 0:
            self.log.error("No datasets found in %s", self.dl4_path)

        obs_ids = observation_config.obs_ids
        if len(obs_ids) == 0:
            # No filtering required based on observation ids
            dl4_file_list = all_dl4_files
        else:
            for dl4_files in all_dl4_files:
                # Assuming a simple nomenclature from gammapy on storing DL4
                # datasets names as pha_obs[OBS_ID].fits or obs_[OBS_ID].fits
                # i.e. a single integer in the filename, being the OBS_ID or
                # the DL4 dataset name.
                obs_num = int(re.findall(r"\d+", dl4_files.name)[0])
                if obs_num in obs_ids:
                    dl4_file_list.append(dl4_files)

        self.log.info("List of DL4 files are: %s", dl4_file_list)

        return dl4_file_list

    def get_dl4_dataset(self, observation_config=None):
        """
        Read the corresponding DL4 dataset with the list of files provided,
        along with the dataset format and stack them in a Datasets object.
        """
        if self.dl4_file:
            datasets = Datasets.read(filename=self.dl4_file)

        elif self.dl4_path:
            dl4_file_list = self.get_dl4_files(observation_config)

            datasets = Datasets()
            for dl4_file in dl4_file_list:
                dataset = DATASET_REGISTRY.get_cls(self.dl4_type)().read(
                    filename=dl4_file, format=self.dl4_dataset.dl4_format
                )
                datasets.append(dataset)

        return datasets

    def get_spectral_energies(self):
        """
        Get the spectral energy information for each Instrument Dataset.
        """
        energy_axes = self.dl4_dataset_info.spectral_energy_range

        if len(energy_axes.axis_custom.edges) > 0:
            energy_bin_edges = get_energy_axis(energy_axes, only_edges=True, custom_range=True)
        else:
            energy_bin_edges = get_energy_axis(
                energy_axes,
                only_edges=True,
            )

        return energy_bin_edges


def get_reco_energy_bins(dataset, en_bins):
    """
    Calculate the total number of fit reco energy bins in the given dataset
    and add to the total value.
    """
    if dataset.mask:
        en_bins += dataset.mask.geom.axes["energy"].nbin
    else:
        en_bins += dataset.counts.geom.axes["energy"].nbin

    return en_bins
