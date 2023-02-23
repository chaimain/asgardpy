"""
Basic classes defining Input Config for DL3 files and some functions to
retrieve the DL3 files information.
"""

import logging
from pathlib import Path

from asgardpy.data.base import BaseConfig, PathType

__all__ = ["InputFilePatterns", "InputConfig", "DL3Files"]

EXPECTED_DL3_RANGE = ["lst-1", "lat", "lat-aux"]

glob_dict_std = {
    "events": "*events.fits*",
    "edisp": "*DRM.fits*",
    "exposure": "*BinnedMap.fits*",
    "xml_model": "*out.xml",
    "psf": "*psf.fits*",
    "diffuse": "gll_iem_v*.fits*",
    "iso": "iso_P8R3_SOURCE_V*_*.txt",
    "dl3": "dl3*fits",
}


# Basic Components for the Input Config
class InputFilePatterns(BaseConfig):
    events: str = "*events.fits*"
    edisp: str = "*DRM.fits*"
    exposure: str = "*BinnedMap.fits*"
    xml_model: str = "*out.xml"
    psf: str = "*psf.fits*"
    diffuse: str = "gll_iem_v*.fits*"
    iso: str = "iso_P8R3_SOURCE_V*_*.txt"
    dl3: str = "dl3*fits"


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
        dl3_path = dir_dict.input_dir
        dl3_type = dir_dict.type
        glob_dict = dir_dict.glob_pattern

        if not log:
            self._set_logging()
        else:
            self.log = log

        if Path(dl3_path).exists():
            self.dl3_path = Path(dl3_path)
        else:
            self.log.error(f"{dl3_path} is not a valid file location")
        self.dl3_type = dl3_type

        if glob_dict is None:
            self.glob_dict = glob_dict_std
        else:
            self.glob_dict = glob_dict
        self._check_dl3_type()

        self.events_files = None
        self.edrm_files = None
        self.xml_files = None
        self.expmap_files = None
        self.psf_files = None
        self.diff_gal_files = None
        self.iso_files = None

        self.xml_f = None
        self.diff_gal_f = None

        self.lat_spectra = None
        self.lat_bute_file = None
        self.lat_ebin_file = None
        self.tag = None

    def _set_logging(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

    def _check_dl3_type(self):
        if self.dl3_type.lower() not in EXPECTED_DL3_RANGE:
            self.log.error(f"{self.dl3_type} is not in the expected range for DL3 files")

    def prepare_lat_files(self, key, file_list):
        """
        Prepare a list of LAT files following a particular key.
        """
        self.tag = key
        # Try to combine LAT and LAT-AUX files
        self.list_dl3_files()
        file_list = self.select_unique_files(self.tag, file_list)

        return file_list

    def select_unique_files(self, key, file_list):
        """
        Select Unique files from all of the provided LAT files, as per the
        given key.
        """
        # Have to make more checks or add conditions on selecting only select
        # files instead from the glob-searched lists.
        if self.dl3_type.lower() == "lat":
            var_list = [
                "events_files",
                "edrm_files",
                "expmap_files",
                "psf_files",
            ]
            file_list["xml_file"] = self.xml_files[0]

        if self.dl3_type.lower() == "lat-aux":
            var_list = [
                "iso_files",
            ]
            if isinstance(self.diff_gal_files, list):
                self.diff_gal_f = self.diff_gal_files[0]
            else:
                self.diff_gal_f = self.diff_gal_files
            file_list["diff_gal_file"] = self.diff_gal_f

        for _v in var_list:
            try:
                filtered = [K for K in getattr(self, _v) if key in str(K)]
                assert len(filtered) == 1
            except Exception:
                self.log.error(
                    f"Variable self.{_v} does not contain one element after filtering by {key}"
                )
            else:
                setattr(self, _v.replace("_files", "_f"), filtered[0])
            file_list[_v.replace("files", "file")] = getattr(self, _v.replace("_files", "_f"))

        return file_list

    def list_dl3_files(self):
        """
        From a given DL3 files path, categorize the different types of DL3
        files, to be used for further analysis.
        """
        if self.dl3_type.lower() == "lat":
            self.events_files = sorted(list(self.dl3_path.glob(self.glob_dict["events"])))
            self.edrm_files = sorted(list(self.dl3_path.glob(self.glob_dict["edisp"])))
            self.xml_files = sorted(list(self.dl3_path.glob(self.glob_dict["xml_model"])))
            self.expmap_files = sorted(list(self.dl3_path.glob(self.glob_dict["exposure"])))
            self.psf_files = sorted(list(self.dl3_path.glob(self.glob_dict["psf"])))

        if self.dl3_type.lower() == "lat-aux":
            self.diff_gal_files = sorted(list(self.dl3_path.glob(self.glob_dict["diffuse"])))
            self.iso_files = sorted(list(self.dl3_path.glob(self.glob_dict["iso"])))

        if self.dl3_type.lower() == "lst-1":
            self.event_files = sorted(list(self.dl3_path.glob(self.glob_dict["dl3"])))
