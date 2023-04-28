"""
Basic classes defining Input Config for DL3 files and some functions to
retrieve the DL3 files information.

Currently supporting files following enrico/fermipy for Fermi-LAT data for 3D
Dataset and DL3 files that follow GADF v0.3 and can be directly read by Gammapy,
for 1D Dataset.
"""

import logging
from pathlib import Path

from asgardpy.base import BaseConfig, PathType

__all__ = ["InputFilePatterns", "InputConfig", "DL3Files"]

EXPECTED_DL3_RANGE = ["gadf-dl3", "lat", "lat-aux"]


# Basic Components for the Input Config
class InputFilePatterns(BaseConfig):
    events: str = "*events.fits*"
    edisp: str = "*DRM.fits*"
    exposure: str = "*BinnedMap.fits*"
    xml_model: str = "*out.xml"
    psf: str = "*psf.fits*"

    dl3: str = "dl3*fits"

    gal_diffuse: str = "gll_iem_v*.fits*"
    iso_diffuse: str = "iso_P8R3_SOURCE_V*_*.txt"


class InputConfig(BaseConfig):
    type: str = "type"
    input_dir: PathType = PathType(".")
    glob_pattern: dict = {}


# Main Classes for I/O
class DL3Files:
    """
    A general class to retrieve information from given DL3 files, along with
    other auxillary files for neighbouring sources, if provided.
    """

    def __init__(self, dir_dict, file_list, log=None):
        if not log:
            self._set_logging()
        else:
            self.log = log

        if Path(dir_dict.input_dir).exists():
            self.dl3_path = Path(dir_dict.input_dir)
        else:
            self.log.error(f"{dir_dict.input_dir} is not a valid file location")

        self.dl3_type = dir_dict.type
        self._check_dl3_type()

        self.glob_dict = dir_dict.glob_pattern

        self.events_files = None
        self.edrm_files = None
        self.xml_files = None
        self.expmap_files = None
        self.psf_files = None
        self.gal_diff_files = None
        self.iso_diff_files = None

        self.xml_f = None
        self.gal_diff_f = None

    def _set_logging(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

    def _check_dl3_type(self):
        if self.dl3_type.lower() not in EXPECTED_DL3_RANGE:
            self.log.error(f"{self.dl3_type} is not in the expected range for DL3 files")

    def prepare_lat_files(self, key, file_list):
        """
        Prepare a list of LAT files following a particular key. If there are no
        distinct key types of files, the value is None.
        """
        # Try to combine LAT and LAT-AUX files
        self.list_dl3_files()
        file_list = self.select_unique_files(key, file_list)

        return file_list

    def list_dl3_files(self):
        """
        From a given DL3 files path, categorize the different types of DL3
        files, to be used for further analysis.

        The dl3_type of 'gadf-dl3' is used for all GADF v0.3 following DL3
        files that can be directly read by Gammapy, for 1D Datasets.
        """
        if self.dl3_type.lower() in ["lat"]:
            self.events_files = sorted(list(self.dl3_path.glob(self.glob_dict["events"])))
            self.edrm_files = sorted(list(self.dl3_path.glob(self.glob_dict["edisp"])))
            self.xml_files = sorted(list(self.dl3_path.glob(self.glob_dict["xml_model"])))
            self.expmap_files = sorted(list(self.dl3_path.glob(self.glob_dict["exposure"])))
            self.psf_files = sorted(list(self.dl3_path.glob(self.glob_dict["psf"])))

        if self.dl3_type.lower() in ["lat-aux"]:
            self.gal_diff_files = sorted(list(self.dl3_path.glob(self.glob_dict["gal_diffuse"])))
            self.iso_diff_files = sorted(list(self.dl3_path.glob(self.glob_dict["iso_diffuse"])))

        if self.dl3_type.lower() in ["gadf-dl3"]:
            self.events_files = sorted(list(self.dl3_path.glob(self.glob_dict["dl3"])))

    def select_unique_files(self, key, file_list):
        """
        Select Unique files from all of the provided LAT files, as per the
        given key. If there are no distinct key types of files, the value is None.
        """
        # Have to make more checks or add conditions on selecting only select
        # files instead from the glob-searched lists.
        if self.dl3_type.lower() in ["lat"]:
            var_list = [
                "events_files",
                "edrm_files",
                "expmap_files",
                "psf_files",
            ]
            file_list["xml_file"] = self.xml_files[0]

        if self.dl3_type.lower() == "lat-aux":
            var_list = []
            if key:
                if "0" not in key:  # For fermipy files, the diffuse files are already unique
                    var_list = [
                        "iso_diff_files",
                    ]
            if isinstance(self.iso_diff_files, list):
                self.iso_gal_f = self.iso_diff_files[0]
            else:
                self.iso_gal_f = self.iso_diff_files
            file_list["iso_diff_file"] = self.iso_gal_f

            if isinstance(self.gal_diff_files, list):
                self.diff_gal_f = self.gal_diff_files[0]
            else:
                self.diff_gal_f = self.gal_diff_files
            file_list["gal_diff_file"] = self.diff_gal_f

        if len(var_list) > 0:
            for _v in var_list:
                try:
                    filtered = [K for K in getattr(self, _v) if key in str(K.name)]
                    assert len(filtered) == 1
                except TypeError:
                    self.log.info("No distinct key provided, selecting the first file in the list")
                    setattr(self, _v.replace("_files", "_f"), getattr(self, _v)[0])
                except Exception:
                    self.log.error(
                        f"Variable self.{_v} does not contain one element after filtering by {key}"
                    )
                else:
                    self.log.info(f"Selecting the file with name containing {key}")
                    setattr(self, _v.replace("_files", "_f"), filtered[0])
                file_list[_v.replace("files", "file")] = getattr(self, _v.replace("_files", "_f"))

        return file_list
