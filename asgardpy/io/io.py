"""
Base I/O functions
"""
import logging
from pathlib import Path

from typing import List
from astropy.io import fits
from gammapy.datasets import FluxPointsDataset
from gammapy.estimators import FluxPoints
from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY, Models
from asgardpy.config import BaseConfig

__all__ = ["DL3Files"]

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
    "spectrum": "Spectrum/SED*.dat",
}


class InputFilePatterns(BaseConfig):
    events: str = "*events.fits*"
    edisp: str = "*DRM.fits*"
    exposure: str = "*BinnedMap.fits*"
    xml_model: str = "*out.xml"
    psf: str = "*psf.fits*"
    diffuse: str = "gll_iem_v*.fits*"
    iso: str = "iso_P8R3_SOURCE_V*_*.txt"
    dl3: str = "dl3*fits"
    spectrum: str = "Spectrum/SED*.dat"


class InputConfig(BaseConfig):
    type: str = "type"
    path: Path = None
    glob_pattern: List[InputFilePatterns] = [InputFilePatterns()]


class IOConfig(BaseConfig):
    type: str = "input_dir"
    input: List[InputConfig] = [InputConfig()]


class DL3Files:
    """
    A general class to retrieve information from given DL3 files, along with
    Models and other auxillary files for neighbouring sources, if provided.
    """

    def __init__(self, dir_dict, source_model):
        dl3_path = dir_dict["path"]
        dl3_type = dir_dict["type"]
        glob_dict = dir_dict["glob_pattern"]

        if Path(dl3_path).exists():
            self.dl3_path = Path(dl3_path)
        else:
            self.log.error("%(dl3_path) is not a valid file")
        self.model = source_model
        self.dl3_type = dl3_type

        if glob_dict is None:
            self.glob_dict = glob_dict_std
        else:
            self.glob_dict = glob_dict

        self._set_logging()
        self._check_model()
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
        self.log.setLevel("%(logging.INFO)")

    def _check_model(self):
        if self.model not in SPECTRAL_MODEL_REGISTRY:
            self.log.error("%(self.model) is not a proper Spectral Model recognized by Gammapy")

    def _check_dl3_type(self):
        if self.dl3_type.lower() not in EXPECTED_DL3_RANGE:
            self.log.error("%(self.dl3_type) is not in the expected range for DL3 files")

    def select_unique_files(self, key):
        """
        Select Unique files from all of the provided LAT files, as per the
        given key.
        """
        if self.dl3_type.lower() == "lat":
            var_list = [
                "events_f",
                "edrm_f",
                "expmap_f",
                "psf_f",
            ]
            self.xml_f = [f for f in self.xml_files if self.model in f][0]

        if self.dl3_type.lower() == "lat-aux":
            var_list = [
                "iso_files",
            ]
            self.diff_gal_f = self.diff_gal_files[0]

        for _v in var_list:
            try:
                filtered = [K for K in getattr(self, _v) if key in K]
                assert len(filtered) == 1
            except Exception:
                self.log.error(
                    "Variable self.%(_v) does not contain one element after filtering by %(key)"
                )
                raise
            else:
                setattr(self, _v.replace("_files", "_f"), filtered[0])

    def list_dl3_files(self):
        """
        From a given DL3 files path, categorize the different types of DL3
        files, to be used for further analysis.
        """
        if self.dl3_type.lower() == "lat":
            self.events_files = sorted(list(self.dl3_path.glob(self.glob_dict["events"])))
            self.log("The list of DL3 event files for LAT selected:", self.events_files)
            self.edrm_files = sorted(list(self.dl3_path.glob(self.glob_dict["edisp"])))
            self.log(
                "The list of Detector Response Matrix files for LAT selected:", self.edrm_files
            )
            self.xml_files = sorted(list(self.dl3_path.glob(self.glob_dict["xml"])))
            self.log("The list of XML files for LAT selected:", self.xml_files)
            self.expmap_files = sorted(list(self.dl3_path.glob(self.glob_dict["exposure"])))
            self.log("The list of Exposure Map files for LAT selected:", self.expmap_files)
            self.psf_files = sorted(list(self.dl3_path.glob(self.glob_dict["psf"])))
            self.log("The list of PSF files for LAT selected:", self.psf_files)

        if self.dl3_type.lower() == "lat-aux":
            self.diff_gal_files = sorted(list(self.dl3_path.glob(self.glob_dict["diffuse"])))
            self.log(
                "The list of Diffuse Galactic sources for LAT-Aux selected:", self.diff_gal_files
            )
            self.iso_files = sorted(list(self.dl3_path.glob(self.glob_dict["iso"])))
            self.log(
                "The list of Isotropic Diffuse model files for LAT-Aux selected:", self.iso_files
            )

        if self.dl3_type.lower() == "lst-1":
            self.event_files = sorted(list(self.dl3_path.glob(self.glob_dict["dl3"])))
            self.log("The list of DL3 files for LST-1 selected:", self.events_files)

    def get_lat_spectra_results(self):
        """
        From the given DL3 files path for LAT files, get the files for the
        spectrum, to be used for further analysis.
        """
        self.lat_bute_file = []
        self.lat_ebin_file = []

        if self.dl3_type.lower() == "lat":
            self.lat_spectra = self.dl3_path.glob(self.glob_dict["spectrum"])
            self.lat_bute_file = [
                K
                for K in self.lat_spectra
                if "cov" not in K and "Ebin" not in K and "ResData" not in K and "fitpars" not in K
            ]
            self.lat_ebin_file = [K for K in self.lat_spectra if "cov" not in K and "Ebin" in K]

    def prepare_lat_files(self, key):
        """
        Prepare a list of LAT files following a particular key.
        """
        self.tag = key
        # Try to combine LAT and LAT-AUX files
        self.list_dl3_files()
        self.get_lat_spectra_results()
        self.select_unique_files(self.tag)


class DL4Files:
    """
    Standard class to read and write DL4 products like the SED, LC and the
    final Fitted Model to common file formats.
    """

    def __init__(self, dl4_path, model, flux_points):
        if Path(dl4_path).exists():
            self.dl4_path = Path(dl4_path)
        else:
            self.log.error("%(dl4_path) does not exist")
        # Check the object type? and take apprpriate measures
        self.model = Models(model)
        self.flux_points = flux_points

        self.flux_from_file = None
        self.model_from_file = None
        self.flux_points_dataset = None

        self._set_logging()

    def _set_logging(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel("%(logging.INFO)")

    def write_model_to_yaml(
        self,
        filename_prefix=None,
        overwrite=True,
        overwrite_templates=False,
        write_covariance=True,
    ):
        """
        Write the whole Models object which maybe a list of SkyModels to YAML
        file to be able to read and use it easily later.
        """
        if filename_prefix is None:
            filename = "spectral_model_dict.yaml"
        else:
            filename = filename_prefix + "_spectral_model_dict.yaml"

        self.model.write(
            self.dl4_path / filename,
            full_output=True,
            overwrite=overwrite,
            write_covariance=write_covariance,
            overwrite_templates=overwrite_templates,
        )

    def write_flux_points_to_fits(
        self, filename=None, sed_type=None, format="gadf-sed", overwrite=True
    ):
        """
        Write the FluxPoints table from a given analysis, to a FITS file, for
        either a given spectral type specified in "sed_type" or for spectral
        and likelihood types.
        """
        if filename is None:
            filename = "flux_points.fits"

        if sed_type is None:
            sed_type = ["dnde", "e2dnde", "likelihood"]

        if not Path(filename).exists():
            flux_file = fits.HDUList([fits.PrimaryHDU()])
        else:
            flux_file = fits.open(filename)

        if isinstance(sed_type, list):
            for sed in sed_type:
                flux_file.append(
                    fits.BinTableHDU(self.flux_points.to_table(sed_type=sed), name=sed)
                )
        else:
            flux_file.append(
                fits.BinTableHDU(self.flux_points.to_table(sed_type=sed_type), name=sed_type)
            )
        flux_file.writeto(self.dl4_path / filename, overwrite=overwrite)
        flux_file.close()

    def write_light_curve_to_fits(self, filename=None, hdu_name=None, overwrite=True):
        """
        Write the FluxPoints table from a given analysis, to a FITS file, for
        either a given spectral type specified in "sed_type" or for spectral
        and likelihood types.
        """
        if filename is None:
            filename = "light_curve_flux.fits"
        if hdu_name is None:
            hdu_name = "LC"

        if not Path(filename).exists():
            flux_file = fits.HDUList([fits.PrimaryHDU()])
        else:
            flux_file = fits.open(filename)

        flux_file.append(
            fits.BinTableHDU(
                self.light_curve_flux.to_table(sed_type="flux", format="lightcurve"), name=hdu_name
            )
        )
        flux_file.writeto(self.dl4_path / filename, overwrite=overwrite)
        flux_file.close()

    def read_model_file(self, filename, only_spectral=True):
        """
        Read from a given YAML file, Models or only the Spectral Model part of
        the Models object.
        """
        self.model_from_file = Models.read(filename)

        if only_spectral:
            self.spectral_model_from_file = self.model_from_file[0].spectral_model

    def read_flux_points(
        self, flux_file, model_file, sed_type="e2dnde",
    ):
        """
        From a given FluxPoints FITS file and Models YAML files, create a
        Gammapy FluxPoints object to be used for later analyses.
        """
        self.flux_from_file = FluxPoints.read(
            filename=flux_file,
            sed_type=sed_type,
            reference_model=self.read_model_file(model_file, only_spectral=True),
        )

    def read_flux_points_dataset(
        self, flux_file, model_file, sed_type="e2dnde",
    ):
        """
        From a given FluxPoints FITS file and Models YAML files, create a
        Gammapy FluxPointsDataset object to be used for later analyses.
        """
        self.flux_from_file = self.read_flux_points(flux_file, sed_type, model_file)

        self.flux_points_dataset = FluxPointsDataset(
            data=self.flux_from_file, models=self.read_model_file(model_file, only_spectral=True)
        )
