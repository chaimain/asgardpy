from astropy.io import fits

from pathlib import Path
import logging
import pickle

from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY


__all__ = []


EXPECTED_DL3_RANGE = ["lst", "lat", "lat-aux"]

class DL3_Files(object):
    def __init__(self, dl3_path, source_model, dl3_type):
        self.dl3_path = Path(dl3_path).exists()
        self.model = source_model
        self.dl3_type = dl3_type
        self._set_logging()
        self._check_model()
        self._check_dl3_type()

    def _set_logging(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

    def _check_model(self):
        if self.model not in SPECTRAL_MODEL_REGISTRY:
            self.log.error(f"{self.model} is not a proper Spectral Model recognized by Gammapy")

    def _check_dl3_type(self):
        if self.dl3_type.lower() not in EXPECTED_DL3_RANGE:
            self.log.error(f"{self.dl3_type} is not in the expected range for DL3 files")

    def select_unique_files(self, key):
        self.unique_name = key
        var_list = [
            "events_files",
            "edrm_files",
            "expmap_files",
            "psf_files",
            "iso_files",
            #'diff_gal_files'
        ]
        for _v in var_list:
            try:
                filtered = [K for K in getattr(self, _v) if key in K]
                assert len(filtered) == 1
            except:
                print(
                    f"Variable self.{_v} does not contain one element after filtering by {key}"
                )
                print(filtered)
                raise
            else:
                setattr(self, _v.replace("_files", "_f"), filtered[0])

        self.xml_f = [f for f in self.xml_files if self.model in f][0]
        self.diff_gal_f = self.diff_gal_files[0]

    def read_dl3_files(self):
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
        if self.dl3_type.lower() == "lat":
            self.lat_spectra = self.dl3_path.glob(f"Spectrum/SED*{self.model}*.dat")
            self.lat_bute_file = [
                K
                for K in self.lat_spectra
                if "cov" not in K
                and "Ebin" not in K
                and "ResData" not in K
                and "fitpars" not in K
            ]
            self.lat_ebin_file = [
                K for K in self.lat_spectra if "cov" not in K and "Ebin" in K
            ]

    def prepare_lat_files(self, key):
        self.tag = key
        # Try to combine LAT and LAT-AUX files
        self.read_dl3_files()
        self.get_lat_spectra_results()
        self.select_unique_files(key)


class DL4_files(object):
    def __init__(self, dl4_path, model, flux_points):
        self.dl4_path = Path(dl4_path).exists()
        self.model = model
        self.flux_points = flux_points

    def write_model_to_dat(self, filename_prefix=None):
        if filename_prefix is None:
            filename = "spectral_model_dict.dat"
        else:
            filename = filename_prefix + "_spectral_model_dict.dat"

        f = open(self.dl4_path / filename, "wb")
        pickle.dump(self.model, f)
        f.close()

    def write_flux_points_to_fits(self, filename_prefix=None):
        if filename_prefix is None:
            filename = "flux_points.fits"
        else:
            filename = filename_prefix + "_flux_points.fits"

        if not Path(filename).exists():
            f = fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.BinTableHDU(
                        self.flux_points.to_table(),
                        name="SED"
                    ),
                ]
            )
            f.writeto(self.dl4_path / filename, overwrite=True)
            f.close()
        else:
            f = fits.open(filename)
            f.append(
                fits.BinTableHDU(
                    self.flux_points.to_table(),
                    name="SED"
                ),
            )
            f.writeto(self.dl4_path / filename, overwrite=True)
            f.close()
