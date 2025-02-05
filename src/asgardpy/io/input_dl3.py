"""
Basic classes defining Input Config for DL3 files and some functions to
retrieve the DL3 files information.

Currently supporting files following enrico/fermipy for Fermi-LAT data for 3D
Dataset and DL3 files that follow GADF v0.3 and can be directly read by Gammapy,
for 1D Dataset.
"""

import logging

from asgardpy.base import BaseConfig, PathType

__all__ = ["InputDL3Config", "DL3Files", "DL3InputFilePatterns"]

EXPECTED_DL3_RANGE = ["gadf-dl3", "lat", "lat-aux", "hawc"]


# Basic Components for the DL3 Input Config
class DL3InputFilePatterns(BaseConfig):
    """
    Config section for list of file patterns to use for fetching relevant DL3
    files.
    """

    events: str = "*events.fits*"
    edisp: str = "*DRM.fits*"
    exposure: str = "*BinnedMap.fits*"
    xml_model: str = "*out.xml"
    psf: str = "*psf.fits*"

    dl3_files: str = "dl3*fits"

    gal_diffuse: str = "gll_iem_v*.fits*"
    iso_diffuse: str = "iso_P8R3_SOURCE_V*_*.txt"

    en_est: str = "*NN*fits.gz"
    transit: str = "TransitsMap*fits.gz"


class InputDL3Config(BaseConfig):
    """
    Config section for main information on getting the relevant DL3 files.
    """

    type: str = "type"
    input_dir: PathType = "None"
    glob_pattern: dict = {}


# Main Classes for I/O
class DL3Files:
    """
    A general class to retrieve information from given DL3 files, along with
    other auxiliary files for neighbouring sources, if provided.
    """

    def __init__(self, dir_dict, log=None):
        if not log:
            self._set_logging()
        else:
            self.log = log

        self.dl3_path = dir_dict.input_dir

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
        self.dl3_index_files = None
        self.transit = None

        self.xml_f = None
        self.gal_diff_f = None

    def _set_logging(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

    def _check_dl3_type(self):
        if self.dl3_type.lower() not in EXPECTED_DL3_RANGE:
            self.log.error("%s is not in the expected range for DL3 files", self.dl3_type)

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
        match self.dl3_type.lower():
            case "lat":
                self.events_files = sorted(list(self.dl3_path.glob(self.glob_dict["events"])))
                self.edrm_files = sorted(list(self.dl3_path.glob(self.glob_dict["edisp"])))
                self.xml_files = sorted(list(self.dl3_path.glob(self.glob_dict["xml_model"])))
                self.expmap_files = sorted(list(self.dl3_path.glob(self.glob_dict["exposure"])))
                self.psf_files = sorted(list(self.dl3_path.glob(self.glob_dict["psf"])))

            case "lat-aux":
                self.gal_diff_files = sorted(list(self.dl3_path.glob(self.glob_dict["gal_diffuse"])))
                self.iso_diff_files = sorted(list(self.dl3_path.glob(self.glob_dict["iso_diffuse"])))

            # case "gadf-dl3":
            ## Is it absolutely essential? we basically rely on the index tables instead of each DL3 file
            # The function is mainly used by LAT files alone at the moment to create a MapDataset from the provided files
            #     if "dl3_files" in self.glob_dict:
            #         self.events_files = sorted(list(self.dl3_path.glob(self.glob_dict["dl3_files"])))
            #     else:
            # For backward compatibility
            #         self.events_files = sorted(list(self.dl3_path.glob(self.glob_dict["dl3"])))

            case "hawc":
                self.transit = sorted(list(self.dl3_path.glob(self.glob_dict["transit"])))
                ## All DL3 index files for a given energy estimator type
                self.dl3_index_files = sorted(list(self.dl3_path.glob(self.glob_dict["en_est"])))

    def select_unique_files(self, key, file_list):
        """
        Select Unique files from all of the provided LAT files, as per the
        given key. If there are no distinct key types of files, the value is None.
        """
        # Have to make more checks or add conditions on selecting only select
        # files instead from the glob-searched lists.
        ## if self.dl3_type.lower() in ["hawc"]:
        # var_
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
                if key is not None:
                    filtered = [K for K in getattr(self, _v) if key in str(K.name)]
                    if len(filtered) == 1:
                        self.log.info("Selecting the file with name containing %s", key)
                        setattr(self, _v.replace("_files", "_f"), filtered[0])
                    else:
                        raise ValueError(
                            "Variable {%s} does not contain one element after filtering by {%s}",
                            getattr(self, _v),
                            key,
                        )
                else:
                    self.log.info("No distinct key provided, selecting the first file in the list")
                    setattr(self, _v.replace("_files", "_f"), getattr(self, _v)[0])

                file_list[_v.replace("files", "file")] = getattr(self, _v.replace("_files", "_f"))

        return file_list
