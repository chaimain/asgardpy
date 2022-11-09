"""
Base I/O functions
"""
import logging
from pathlib import Path

from astropy.io import fits
from gammapy.datasets import FluxPointsDataset
from gammapy.estimators import FluxPoints
from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY, Models

# __all__ = []


EXPECTED_DL3_RANGE = ["lst", "lat", "lat-aux"]


class DL3Files:
    """
    A general class to retrieve information from given DL3 files, along with
    Models and other auxillary files for neighbouring sources, if provided.
    """
    def __init__(self, dl3_path, source_model, dl3_type):

        if Path(dl3_path).exists():
            self.dl3_path = Path(dl3_path)
        else:
            self.log.error("%(dl3_path) is not a valid file")
        self.model = source_model
        self.dl3_type = dl3_type
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
        # self.unique_name = key
        var_list = [
            "events_files",
            "edrm_files",
            "expmap_files",
            "psf_files",
            "iso_files",
            # 'diff_gal_files'
        ]
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

        self.xml_f = [f for f in self.xml_files if self.model in f][0]
        self.diff_gal_f = self.diff_gal_files[0]

    def list_dl3_files(self):
        """
        From a given DL3 files path, categorize the different types of DL3
        files, to be used for further analysis.
        """
        if self.dl3_type.lower() == "lat":
            self.events_files = self.dl3_path.glob("*_MkTime.fits*")
            self.edrm_files = self.dl3_path.glob(f"*{self.model}_*eDRM.fits*")
            self.xml_files = self.dl3_path.glob("*_out.xml")
            self.expmap_files = self.dl3_path.glob("*_BinnedMap.fits*")
            self.psf_files = self.dl3_path.glob("*_psf.fits*")

        if self.dl3_type.lower() == "lat-aux":
            self.diff_gal_files = self.dl3_path.glob("gll_iem_v07.fits*")
            self.iso_files = self.dl3_path.glob("iso_P8R3_SOURCE_V3_*.txt")

    def get_lat_spectra_results(self):
        """
        From the given DL3 files path for LAT files, get the files for the
        spectrum, to be used for further analysis.
        """
        if self.dl3_type.lower() == "lat":
            self.lat_spectra = self.dl3_path.glob(f"Spectrum/SED*{self.model}*.dat")
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
        self.model = model
        self.flux_points = flux_points

        self.flux_from_file = None
        self.model_from_file = None
        self.flux_points_dataset = None

        self._set_logging()

    def _set_logging(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel("%(logging.INFO)")

    def write_model_to_yaml(self, filename_prefix=None, overwrite=True):
        """
        Write the whole Models object which maybe a list of SkyModels to YAML
        file to be able to read and use it easily later.
        """
        if filename_prefix is None:
            filename = "spectral_model_dict.yaml"
        else:
            filename = filename_prefix + "_spectral_model_dict.yaml"

        self.model.write(self.dl4_path / filename, overwrite=overwrite)

    def write_flux_points_to_fits(self, filename_prefix=None, overwrite=True):
        """
        Write the FluxPoints object from a given analysis, to a FITS file.
        """
        if filename_prefix is None:
            filename = "flux_points.fits"
        else:
            filename = filename_prefix + "_flux_points.fits"

        if not Path(filename).exists():
            flux_file = fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.BinTableHDU(
                        self.flux_points.to_table(),
                        name="SED"
                    ),
                ]
            )
            flux_file.writeto(self.dl4_path / filename, overwrite=overwrite)
            flux_file.close()
        else:
            flux_file = fits.open(filename)
            flux_file.append(fits.BinTableHDU(
                self.flux_points.to_table(), name="SED")
            )
            flux_file.writeto(self.dl4_path / filename, overwrite=overwrite)
            flux_file.close()

    def read_flux_points_dataset(self, flux_file, model_file):
        """
        From a given FluxPoints FITS file and Models YAML files, create a
        Gammapy FluxPointsDataset object to be used for later analyses.
        """
        self.flux_from_file = FluxPoints.read(flux_file)
        self.model_from_file = Models.read(model_file)

        self.flux_points_dataset = FluxPointsDataset(
            data=self.flux_from_file,
            models=self.model_from_file
        )
