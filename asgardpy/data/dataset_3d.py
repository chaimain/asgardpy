"""
Generating 3D Datasets from given Instrument DL3 data
"""

import gzip
import logging

import numpy as np
import xmltodict
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

# from gammapy.analysis import Analysis, AnalysisConfig - no support for DL3 with RAD_MAX
from gammapy.data import EventList
from gammapy.datasets import MapDataset
from gammapy.irf import EDispKernel, EDispKernelMap, PSFMap
from gammapy.maps import Map, MapAxis
from gammapy.modeling import Parameters
from gammapy.modeling.models import (
    SPECTRAL_MODEL_REGISTRY,
    EBLAbsorptionNormSpectralModel,
    LogParabolaSpectralModel,
    Models,
    PointSpatialModel,
    PowerLawNormSpectralModel,
    PowerLawSpectralModel,
    SkyModel,
    TemplateSpatialModel,
    create_fermi_isotropic_diffuse_model,
)
from regions import CircleSkyRegion

from asgardpy.io.io import DL3Files

__all__ = ["Dataset3D", "Dataset3DIO", "Dataset3DInfo"]

log = logging.getLogger(__name__)


class Dataset3DIO(DL3Files):
    """
    Read the DL3 files of 3D datasets.
    """

    def __init__(self, config, instrument_idx=0, key_idx=0):
        self.config = config
        self.config_3d_dataset = self.config["Dataset3D"]["Instruments"][instrument_idx]
        self.config_3d_dataset_io = self.config_3d_dataset["IO"]
        self.key_name = self.config_3d_dataset_info["key"][key_idx]

    def get_base_objects(self, dl3_path, model, dl3_type, key):
        """
        For a DL3 files type and tag of the 'mode of observations' - FRONT or
        BACK, read the files to appropriate Object type for further analysis.
        """
        temp = DL3Files(self, dl3_path, model, dl3_type)
        temp.prepare_lat_files(key)

        if dl3_type.lower() == "lat":
            self.exposure = Map.read(self.expmap_f)
            self.psf = PSFMap.read(self.psf_f, format="gtpsf")
            self.drmap = fits.open(self.edrm_f)
            self.edisp_kernel = self.set_energy_dispersion_matrix()
            return [self.exposure, self.psf, self.drmap, self.edisp_kernel]

        elif dl3_type.lower() == "lat-aux":
            self.diff_gal = Map.read(self.diff_gal_f)
            self.diff_iso = create_fermi_isotropic_diffuse_model(
                filename=self.iso_f, interp_kwargs={"fill_value": None}
            )
            self.diff_iso.name = f"{self.diff_iso.name}-{self.key_name}"

            # Parameters' limits generalization?
            self.diff_iso.spectral_model.model1.parameters[0].min = 0.001
            self.diff_iso.spectral_model.model1.parameters[0].max = 10
            self.diff_iso.spectral_model.model2.parameters[0].min = 0
            self.diff_iso.spectral_model.model2.parameters[0].max = 10

            return [self.diff_gal, self.diff_iso]
        else:
            return []

    def set_energy_axes(self):
        """
        Get the energy axes from the given DRM file.
        """
        energy_lo = self.drmap["DRM"].data["ENERG_LO"] * u.MeV
        energy_hi = self.drmap["DRM"].data["ENERG_HI"] * u.MeV

        energy_axis = MapAxis.from_energy_edges(np.append(energy_lo[0], energy_hi))
        energy_axis_true = energy_axis.copy(name="energy_true")

        return energy_axis, energy_axis_true

    def set_energy_dispersion_matrix(self):
        """
        Generate the Energy Dispersion Kernel from the file.
        """
        self.energy_axis, self.energy_axis_true = self.set_energy_axes()
        drm = self.drmap["DRM"].data["MATRIX"]
        drm_matrix = np.array(list(drm))

        self.edisp_kernel = EDispKernel(
            axes=[self.energy_axis_true, self.energy_axis], data=drm_matrix
        )

    def load_events(self):
        """
        Loading the events files for the specific "Key" and saving them to a
        dummy fits file if the original files are gzipped.
        """
        # Create a local file (important if gzipped, as sometimes it fails to read)
        try:
            with gzip.open(self.events_f) as gzfile:
                with open("temp_events.fits", "wb") as file:
                    unzipped_file = gzip.decompress(gzfile.read())
                    file.write(unzipped_file)

            self.event_fits = fits.open("temp_events.fits")
            self.events = EventList.read("temp_events.fits")
        except Exception:
            self.event_fits = fits.open(self.events_f)
            self.events = EventList.read(self.events_f)

    def get_src_skycoord(self):
        """
        Get the source skycoord from the events file.
        """
        try:
            dsval2 = self.event_fits[1].header["DSVAL2"]
            ra_pos, dec_pos = [float(k) for k in dsval2.split("(")[1].split(",")[0:2]]
        except IndexError:
            history = str(self.event_fits[1].header["HISTORY"])
            ra_pos, dec_pos = (
                history.split("angsep(RA,DEC,")[1].replace("\n", "").split(")")[0].split(",")
            )

        self.src_pos = SkyCoord(ra_pos, dec_pos, unit="deg", frame="fk5").icrs

    def get_list_objects(self, aux_path, lp_is_intrinsic=False):
        """
        Read from the XML file to enlist the various objects and get their
        SkyModels
        """
        self.list_sources = []

        with open(self.xml_f) as f:
            data = xmltodict.parse(f.read())["source_library"]["source"]
            self.list_of_sources_final = [src["@name"] for src in data]

        for src in data:
            source_name = src["@name"]
            if source_name == "IsoDiffModel":
                source = self.diff_iso
            elif source_name == "GalDiffModel":
                source = self.create_gal_diffuse_skymodel()
            else:
                source = self.create_source_skymodel(src, aux_path, lp_is_intrinsic)

            self.list_sources.append(source)

    def create_source_skymodel(self, src, aux_path, lp_is_intrinsic=False):
        """
        Build SkyModels from a given list of LAT files
        """
        source_name = src["@name"]
        spectrum_type = src["spectrum"]["@type"].split("EblAtten::")[-1]
        spectrum = src["spectrum"]["parameter"]
        spatial_pars = src["spatialModel"]["parameter"]

        source_name_red = source_name.replace("_", "").replace(" ", "")
        target_red = self.config["Target_source"]["source_name"].replace("_", "").replace(" ", "")

        # Check if target_source file exists
        if source_name_red == target_red:
            source_name = self.config["Target_source"]["source_name"]
            is_src_target = True
            self.log.debug("Detected target source")
        else:
            is_src_target = False

        for spec in spectrum:
            if spec["@name"] not in ["GalDiffModel", "IsoDiffModel"]:
                if spectrum_type == "PLSuperExpCutoff":
                    spectrum_type_final = "ExpCutoffPowerLawSpectralModel"
                elif spectrum_type == "PLSuperExpCutoff4":
                    spectrum_type_final = "SuperExpCutoffPowerLaw4FGLDR3SpectralModel"
                else:
                    spectrum_type_final = f"{spectrum_type}SpectralModel"

                spec_model = SPECTRAL_MODEL_REGISTRY.get_cls(spectrum_type_final)
                if spectrum_type == "LogParabola" and "EblAtten" in src["spectrum"]["@type"]:
                    if lp_is_intrinsic:
                        ebl_atten_pl = False
                        spec_model = LogParabolaSpectralModel()
                    else:
                        ebl_atten_pl = True
                        spec_model = PowerLawSpectralModel()

                params_list = self.xml_to_gammapy_model_params(
                    spectrum,
                    spectrum_type,
                    is_target=is_src_target,
                    keep_sign=ebl_atten_pl,
                    lp_is_intrinsic=lp_is_intrinsic,
                )
                spec_model.from_parameters(params_list)

        ebl_absorption = self.config["Target_model"]["spectral"]["ebl_abs"]
        ebl_absorption_included = len(ebl_absorption) > 0

        if is_src_target and ebl_absorption_included:
            ebl_model = ebl_absorption["model_name"]
            ebl_spectral_model = EBLAbsorptionNormSpectralModel.read_builtin(
                ebl_model, redshift=self.config["Target_source"]["redshift"]
            )
            spec_model = spec_model * ebl_spectral_model

        if src["spatialModel"]["@type"] == "SkyDirFunction":
            fk5_frame = SkyCoord(
                lon=f"{spatial_pars[0]['@value']} deg",
                lat=f"{spatial_pars[1]['@value']} deg",
                frame="fk5",
            )
            gal_frame = fk5_frame.transform_to("galactic")
            spatial_model = PointSpatialModel(
                lon_0=gal_frame.lon, lat_0=gal_frame.lat, frame="galactic"
            )
        elif src["spatialModel"]["@type"] == "SpatialMap":
            file_name = src["spatialModel"]["@file"].split("/")[-1]
            file_path = aux_path / f"Templates/{file_name}"

            spatial_map = Map.read(file_path)
            spatial_map = spatial_map.copy(unit="sr^-1")

            spatial_model = TemplateSpatialModel(spatial_map, filename=file_path)

        spatial_model.freeze()
        source_sky_model = SkyModel(
            spectral_model=spec_model,
            spatial_model=spatial_model,
            name=source_name,
        )

        return source_sky_model

    def xml_to_gammapy_model_params(
        self, params, spectrum_type, is_target=False, keep_sign=False, lp_is_intrinsic=False
    ):
        """
        Make some basic conversions on the spectral model parameters
        """
        new_model = []
        for par in params:
            new_par = {}
            # For EBL Attenuated Power Law, it is taken with LogParabola model
            # and turning beta value off
            if lp_is_intrinsic and par["@name"] == "beta":
                continue

            for k in par.keys():
                # Replacing the "@par_name" information of each parameter without the "@"
                if k != "@free":
                    new_par[k[1:].lower()] = par[k]
                else:
                    new_par["frozen"] = (par[k] == "0") and not is_target
                new_par["unit"] = ""
                new_par["is_norm"] = False

                # Using the nomenclature as used in Gammapy
                # Make scale = 1, by multiplying it to the value, min and max?
                if par["@name"].lower() in ["norm", "prefactor", "integral"]:
                    new_par["name"] = "amplitude"
                    new_par["unit"] = "cm-2 s-1 MeV-1"
                    new_par["is_norm"] = True
                if par["@name"].lower() in ["scale", "eb"]:
                    new_par["name"] = "reference"
                    new_par["frozen"] = par[k] == "0"
                if par["@name"].lower() in ["breakvalue"]:
                    new_par["name"] = "ebreak"
                if par["@name"].lower() in ["lowerlimit"]:
                    new_par["name"] = "emin"
                if par["@name"].lower() in ["upperlimit"]:
                    new_par["name"] = "emax"
                if new_par["name"] in ["reference", "ebreak", "emin", "emax"]:
                    new_par["unit"] = "MeV"
                if par["@name"].lower() in ["cutoff"]:
                    new_par["name"] = "lambda_"
                    new_par["value"] = 1.0 / new_par["value"]
                    new_par["min"] = 1.0 / new_par["min"]
                    new_par["max"] = 1.0 / new_par["max"]
                    new_par["unit"] = "MeV-1"

                # More modifications:
                if new_par["name"] == "index" and not keep_sign:
                    # Other than EBL Attenuated Power Law
                    new_par["value"] *= -1
                    new_par["min"] *= -1
                    new_par["max"] *= -1

            new_model.append(Parameters().update_from_dict(new_par))

        return new_model

    def read_to_objects(self):
        """ """
        model = self.config["Target_model"]["spectral"]["model_name"]
        lp_is_intrinsic = model == "LogParabola"

        # Multiple input paths
        inp_cfg = self.config_3d_dataset_io["input_dir"]
        dl3_type_1 = inp_cfg[0]["type"]
        dl3_type_2 = inp_cfg[1]["type"]  # Optional or something more general?
        dl3_path_1 = inp_cfg[0]["path"]
        dl3_path_2 = inp_cfg[1]["path"]
        # For each key, get all the base objects.
        self.exposure, self.psf, self.drmap, self.edisp_kernel = self.get_base_objects(
            dl3_path_1, model, dl3_type_1, self.key_name
        )
        self.diff_gal, self.diff_iso = self.get_base_objects(
            dl3_path_2, model, dl3_type_2, self.key_name
        )
        self.get_list_objects(dl3_path_2, lp_is_intrinsic)


class Dataset3DInfo(Dataset3DIO):
    """
    Prepare standard data reduction using the parameters passed in the config
    for 3D datasets.
    """

    def __init__(self, config, instrument_idx=0, key_idx=0):
        self.config = config
        self.config_3d_dataset = self.config["Dataset3D"]["Instruments"][instrument_idx]
        self.config_3d_dataset_info = self.config_3d_dataset["DatasetInfo"]
        self.exclusion_regions = []

        io = Dataset3DIO()
        io.read_to_objects()
        io.set_energy_dispersion_matrix()
        io.load_events()
        io.get_src_skycoord()

        self._counts_map()
        self._set_edisp_interpolator()
        self._set_exposure_interpolator()
        self._generate_diffuse_background_cutout()

    def _counts_map(self):
        """ """
        self.counts_map = Map.create(
            skydir=self.src_pos,
            npix=(self.exposure.geom.npix[0][0], self.exposure.geom.npix[1][0]),
            proj="TAN",
            frame="fk5",
            binsz=self.exposure.geom.pixel_scales[0],
            axes=[self.energy_axis],
            dtype=float,
        )
        self.counts_map.fill_by_coord({"skycoord": self.events.radec, "energy": self.events.energy})

    def _generate_diffuse_background_cutout(self):
        """
        Doing a cutout of the Diffuse background model with respect to the
        counts map geom, may improve fitting speed.

        The Template Spatial Model is without normalization currently.
        """
        self.diffuse_cutout = self.diff_gal.cutout(
            self.counts_map.geom.center_skydir, self.counts_map.geom.width[0]
        )

        self.template_diffuse = TemplateSpatialModel(self.diffuse_cutout, normalize=False)

        self.diff_gal_cutout = SkyModel(
            spectral_model=PowerLawNormSpectralModel(),
            spatial_model=self.template_diffuse,
            name="diffuse-iem",
        )
        self.diff_gal_cutout.parameters["norm"].min = 0
        self.diff_gal_cutout.parameters["norm"].max = 10
        self.diff_gal_cutout.parameters["norm"].frozen = False

    def _set_edisp_interpolator(self):
        """
        Get Energy Dispersion Kernel interpolated along true and reconstructed
        energy of the real counts.
        """
        axis_reco = MapAxis.from_edges(
            self.counts_map.geom.axes["energy"].edges,
            name="energy",
            unit="MeV",
            interp="log",
        )
        axis_true = axis_reco.copy(name="energy_true")
        energy_reco, energy_true = np.meshgrid(axis_true.center, axis_reco.center)

        drm_interp = self.edisp_kernel.evaluate(
            "linear", **{"energy": energy_reco, "energy_true": energy_true}
        )
        self.edisp_interp_kernel = EDispKernel(
            axes=[axis_true, axis_reco], data=np.asarray(drm_interp)
        )

    def _set_exposure_interpolator(self):
        """
        Set Exposure interpolated along energy axis of real counts.
        """
        self.exposure_interp = self.exposure.interp_to_geom(self.counts_map.geom.as_energy_true)

    def _create_exclusion_mask(self):
        """
        Generate an Exclusion Mask for the final MapDataset
        """
        excluded_geom = self.counts_map.geom.copy()

        if len(self.exclusion_regions) == 0:
            self.log.info("Creating empty/dummy exclusion region")
            pos = SkyCoord(0, 90, unit="deg")
            exclusion_region = CircleSkyRegion(pos, 0.00001 * u.deg)
            self.exclusion_mask = ~excluded_geom.region_mask([exclusion_region])
        else:
            self.log.info("Creating exclusion region")
            self.exclusion_mask = ~excluded_geom.region_mask(self.exclusion_regions)

    def add_source_to_exclusion_region(self, source_pos=None, radius=0.1 * u.deg):
        """
        Add to an existing exclusion_regions list, the source region.
        """
        if source_pos is not None:
            exclusion_region = CircleSkyRegion(
                center=source_pos.galactic,  # Generalize frame or ask from user.
                radius=radius,
            )
        else:
            sky_pos = self.config["Target_source"]["sky_position"]
            source_pos = SkyCoord.from_name(
                ra=u.quantity(sky_pos["ra"]), dec=u.quantity(sky_pos["dec"])
            )
            exclusion_region = CircleSkyRegion(
                center=source_pos.galactic,
                radius=radius,
            )
        self.exclusion_regions.append(exclusion_region)


class Dataset3D(Dataset3DInfo):
    """
    Main class to generate the 3D Dataset for a given Instrument information.
    """

    def __init__(self):
        self.dataset = None
        self.instrument_idx = 0
        self.key_idx = 0
        info = Dataset3DInfo(self.config, self.instrument_idx, self.key_idx)
        info()

    def create_dataset(self):
        """
        Generate MapDataset for the given Instrument files.
        """
        try:
            mask_safe = self.exclusion_mask
        except Exception:
            mask_bool = np.zeros(self.counts_map.geom.data_shape).astype("bool")
            mask_safe = Map.from_geom(self.counts_map.geom, mask_bool)
            mask_safe.data = np.asarray(mask_safe.data == 0, dtype=bool)

        self.dataset = MapDataset(
            models=Models(self.list_sources),
            counts=self.counts_map,
            exposure=self.exposure_interp,
            psf=self.psf,
            edisp=EDispKernelMap.from_edisp_kernel(self.edisp_interp_kernel),
            mask_safe=mask_safe,
            name="Fermi-LAT_{}".format(self.key_name),
        )

    def get_source_pos_from_3d_dataset(self):
        """
        Introduce the source coordinates from 3D dataset to 1D dataset.
        """
        for src in self.list_sources:
            if src.name == self.config["Target_source"]["source_name"]:
                source_position_from_3d = SkyCoord(
                    lon=src.spatial_model["lon_0"], lat=src.spatial_model["lat_0"], frame="galactic"
                ).icrs
                return source_position_from_3d
            else:
                return None
