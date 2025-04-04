"""
Basic classes defining Input Config for DL4 files and some functions to
retrieve information for the DL4 to DL5 processes.
"""

import logging
import re
from enum import Enum
from pathlib import Path

from gammapy.datasets import DATASET_REGISTRY, Datasets
from gammapy.modeling.models import Models

from asgardpy.base.base import BaseConfig, PathType
from asgardpy.base.geom import MapAxesConfig, get_energy_axis

__all__ = [
    "InputDL4Config",
    "DL4Files",
    "DL4InputFilePatterns",
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


class DL4InputFilePatterns(BaseConfig):
    """
    Config section for list of file patterns to use for fetching relevant DL4
    files.
    """

    dl4_files: str = "pha*.fits*"
    dl4_model_files: str = "model*yaml"


class InputDL4Config(BaseConfig):
    """
    Config section for main information on getting the relevant DL4 files.
    """

    type: DatasetTypeEnum = DatasetTypeEnum.MapDataset
    input_dir: PathType = "None"
    # Can be OGIP format (Stacked or unstacked obs) or fits format (stacked obs)
    glob_pattern: dict = {}
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
        self.dl4_model = None

        if Path(self.dl4_dataset.input_dir).is_file():
            self.dl4_file = Path(self.dl4_dataset.input_dir)
        else:
            self.dl4_path = Path(self.dl4_dataset.input_dir)

        if not log:
            self._set_logging()
        else:
            self.log = log

    def _set_logging(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

    def fetch_dl4_files_by_filenames(self, all_dl4_files, obs_ids):
        """
        Assuming a simple nomenclature from gammapy on storing DL4 datasets
        names as pha_obs[OBS_ID].fits or obs_[OBS_ID].fits i.e. a single integer
        in the filename, being the OBS_ID or the DL4 dataset name.
        """
        dl4_file_list = []
        for dl4_files in all_dl4_files:
            obs_num = int(re.findall(r"\d+", dl4_files.name)[0])
            if obs_num in obs_ids:
                dl4_file_list.append(dl4_files)
        return dl4_file_list

    def read_dl4_file(self, filename):
        """
        Read a single file, which may be serialized in FITS or yaml format.
        """
        if str(filename)[-4:] == "yaml":
            return Datasets.read(filename=filename)
        elif str(filename)[-4:] in ["fits", "s.gz"]:
            dataset_ = DATASET_REGISTRY.get_cls(self.dl4_type)().read(
                filename=filename, format=self.dl4_dataset.dl4_format
            )
            return Datasets(dataset_)
        else:
            return None

    def get_dl4_files(self, observation_config):
        """
        Fetch the required DL4 files from the given directory path, file glob
        search and possible list of observation ids to select the dataset files
        from the full list in the directory.

        If Model files are also given, fetch them as well
        """
        dl4_model_files = []

        all_dl4_files = sorted(list(self.dl4_path.glob(self.dl4_dataset.glob_pattern["dl4_files"])))
        # Get model files as well
        if "dl4_model_files" in self.dl4_dataset.glob_pattern.keys():
            dl4_model_files = sorted(list(self.dl4_path.glob(self.dl4_dataset.glob_pattern["dl4_model_files"])))

        if len(all_dl4_files) == 0:
            self.log.error("No datasets found in %s", self.dl4_path)

        obs_ids = observation_config.obs_ids
        if len(obs_ids) == 0:
            # No filtering required based on observation ids
            dl4_file_list = all_dl4_files
        else:
            dl4_file_list = self.fetch_dl4_files_by_filenames(all_dl4_files, obs_ids)

        self.log.info("List of DL4 files are: %s", dl4_file_list)

        return dl4_file_list, dl4_model_files

    def get_dl4_dataset(self, observation_config=None):
        """
        Read the corresponding DL4 dataset with the list of files provided,
        along with the dataset format and stack them in a Datasets object.
        """
        if self.dl4_file:
            datasets = Datasets.read(filename=self.dl4_file)

        elif self.dl4_path:
            dl4_file_list, dl4_model_files = self.get_dl4_files(observation_config)

            if len(dl4_model_files) == 0:
                datasets = Datasets()
                for dl4_file in dl4_file_list:
                    dataset = self.read_dl4_file(dl4_file)
                    datasets.append(dataset[0])
            else:
                # Assuming a single DL4 file and model
                datasets = self.read_dl4_file(dl4_file_list[0])
                datasets.models = Models.read(dl4_model_files[0])

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
    en_bins += dataset.mask.geom.axes["energy"].nbin

    return en_bins
