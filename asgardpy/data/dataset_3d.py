"""
Generating 3D Datasets from given Instrument DL3 data
"""

import gzip
import logging
from typing import List

import numpy as np
import xmltodict
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

# from gammapy.analysis import Analysis, AnalysisConfig - no support for DL3 with RAD_MAX
from gammapy.data import EventList
from gammapy.datasets import MapDataset, Datasets
from gammapy.irf import EDispKernel, EDispKernelMap, PSFMap
from gammapy.makers import MapDatasetMaker
from gammapy.maps import Map, MapAxis
from gammapy.modeling import Parameter, Parameters
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

from asgardpy.data.base import AnalysisStepBase, BaseConfig
from asgardpy.data.geom import SpatialCircleConfig
from asgardpy.data.reduction import (
    BackgroundConfig,
    MapSelectionEnum,
    ReductionTypeEnum,
    SafeMaskConfig,
)
from asgardpy.io import DL3Files, InputConfig

__all__ = [
    "Dataset3DInfoConfig",
    "Dataset3DBaseConfig",
    "Dataset3DConfig",
    "Dataset3DGeneration",
    "Datasets3DAnalysisStep",
]

log = logging.getLogger(__name__)


class Dataset3DInfoConfig(BaseConfig):
    name: str = "dataset-name"
    key: List = []
    map_selection: List[MapSelectionEnum] = MapDatasetMaker.available_selection
    background: BackgroundConfig = BackgroundConfig()
    safe_mask: SafeMaskConfig = SafeMaskConfig()
    on_region: SpatialCircleConfig = SpatialCircleConfig()
    containment_correction: bool = True


class Dataset3DBaseConfig(BaseConfig):
    name: str = "Instrument-name"
    io: List[InputConfig] = [InputConfig()]
    dataset_info: Dataset3DInfoConfig = Dataset3DInfoConfig()


class Dataset3DConfig(BaseConfig):
    type: ReductionTypeEnum = ReductionTypeEnum.cube
    instruments: List[Dataset3DBaseConfig] = [Dataset3DBaseConfig()]


class Datasets3DAnalysisStep(AnalysisStepBase):
    """
    Main class to generate the 3D Dataset for a given Instrument information.
    """

    tag = "datasets-3d"

    def _run(self):
        instruments_list = self.config.dataset3d.instruments
        self.log.info(f"{len(instruments_list)} number of 3D Datasets given")

        datasets_3d_final = Datasets()

        for i in np.arange(len(instruments_list)):
            self.config_3d_dataset = instruments_list[i]
            key_names = self.config_3d_dataset.dataset_info.key
            self.log.info(f"The different keys used: {key_names}")

            for key in key_names:
                generate_3d_dataset = Dataset3DGeneration(
                    self.config_3d_dataset, self.config.target, key
                )
                dataset = generate_3d_dataset.run()
                datasets_3d_final.append(dataset)

        return datasets_3d_final


class Dataset3DGeneration:
    """
    Separate class on 3D dataset creation based on the config or
    AsgardpyConfig information provided on the 3D dataset and the target source.

    Runs the following steps:
    1. Read the DL3 files of 3D datasets into gammapy readable objects.
    2. Prepare standard data reduction using the parameters passed in the config
    for 3D datasets.
    3. Generate the final dataset.
    """

    def __init__(self, config_3d_dataset, config_target, key_name):
        self.config_3d_dataset_io = config_3d_dataset.io
        self.key_name = key_name
        self.config_target = config_target
        self.model = self.config_target.components.spectral
        self.exclusion_regions = []
        self.target_full_model = None

    def run(self):
        file_list = self.read_to_objects(self.model, self.key_name)
        self.set_energy_dispersion_matrix()
        self.load_events(file_list["events_file"])
        self.get_src_skycoord()
        self._counts_map()
        self._create_exclusion_mask()
        self.add_source_to_exclusion_region()
        self._set_edisp_interpolator()
        self._set_exposure_interpolator()
        self._generate_diffuse_background_cutout()
        dataset = self.generate_dataset(self.key_name)

        return dataset

    def read_to_objects(self, model, key_name):
        """
        For each key type of files, read the files to get the required
        Gammapy objects for further analyses.
        """
        lp_is_intrinsic = model == "LogParabola"
        file_list = {}

        for cfg in self.config_3d_dataset_io:
            if cfg.type == "lat":
                file_list, [
                    self.exposure,
                    self.psf,
                    self.drmap,
                    self.edisp_kernel,
                ] = self.get_base_objects(cfg, model, key_name, cfg.type, file_list)
            if cfg.type == "lat-aux":
                file_list, [self.diff_gal, self.diff_iso] = self.get_base_objects(
                    cfg, model, key_name, cfg.type, file_list
                )
                self.get_list_objects(cfg.input_dir, file_list["xml_file"], lp_is_intrinsic)

        return file_list

    def get_source_pos_from_3d_dataset(self):
        """
        Introduce the source coordinates from 3D dataset to 1D dataset.
        Need to generalize this as well for all datasets.
        """
        source_position_from_3d = None
        for src in self.list_sources:
            if src.name == self.config_target.source_name:
                source_position_from_3d = src.spatial_model.position.icrs

        return source_position_from_3d

    def get_base_objects(self, dl3_dir_dict, model, key, dl3_type, file_list):
        """
        For a DL3 files type and tag of the 'mode of observations' - FRONT or
        BACK, read the files to appropriate Object type for further analysis.
        """
        temp = DL3Files(dl3_dir_dict, model, file_list)
        file_list = temp.prepare_lat_files(key, file_list)

        if dl3_type.lower() == "lat":
            self.exposure = Map.read(file_list["expmap_file"])
            self.psf = PSFMap.read(file_list["psf_file"], format="gtpsf")
            self.drmap = fits.open(file_list["edrm_file"])
            self.edisp_kernel = self.set_energy_dispersion_matrix()
            return file_list, [self.exposure, self.psf, self.drmap, self.edisp_kernel]

        elif dl3_type.lower() == "lat-aux":
            self.diff_gal = Map.read(file_list["diff_gal_file"])
            self.diff_iso = self.create_iso_diffuse_skymodel(file_list["iso_file"], key)

            return file_list, [self.diff_gal, self.diff_iso]
        else:
            return file_list, []

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

    def load_events(self, events_file):
        """
        Loading the events files for the specific "Key" and saving them to a
        dummy fits file if the original files are gzipped.
        """
        # Create a local file (important if gzipped, as sometimes it fails to read)
        # Check again the valididty of gzipping files, and also on the use
        # of EventList, instead of other Gammapy object
        try:
            with gzip.open(events_file) as gzfile:
                with open("temp_events.fits", "wb") as file:
                    unzipped_file = gzip.decompress(gzfile.read())
                    file.write(unzipped_file)

            self.event_fits = fits.open("temp_events.fits")
            self.events = EventList.read("temp_events.fits")
        except Exception:
            self.event_fits = fits.open(events_file)
            self.events = EventList.read(events_file)

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

        self.src_pos = SkyCoord(ra_pos, dec_pos, unit="deg", frame="fk5")

    def get_list_objects(self, aux_path, xml_file, lp_is_intrinsic=False):
        """
        Read from the XML file to enlist the various objects and get their
        SkyModels
        """
        self.list_sources = []

        with open(xml_file) as f:
            data = xmltodict.parse(f.read())["source_library"]["source"]
            self.list_of_sources_final = [src["@name"] for src in data]

        for src in data:
            source_name = src["@name"]
            if source_name == "IsoDiffModel":
                source = self.diff_iso
            elif source_name == "GalDiffModel":
                source = self.create_gal_diffuse_skymodel(self.diff_gal)
            else:
                source, is_target_source = self.create_source_skymodel(src, aux_path, lp_is_intrinsic)
            if is_target_source:
                self.target_full_model = source
            self.list_sources.append(source)

    def create_iso_diffuse_skymodel(self, iso_file, key):
        """
        Create a SkyModel of the Fermi Isotropic Diffuse Model.
        """
        diff_iso = create_fermi_isotropic_diffuse_model(
            filename=iso_file, interp_kwargs={"fill_value": None}
        )
        diff_iso._name = f"{diff_iso.name}-{key}"

        # Parameters' limits generalization?
        diff_iso.spectral_model.model1.parameters[0].min = 0.001
        diff_iso.spectral_model.model1.parameters[0].max = 10
        diff_iso.spectral_model.model2.parameters[0].min = 0
        diff_iso.spectral_model.model2.parameters[0].max = 10

        return diff_iso

    def create_gal_diffuse_skymodel(self, diff_gal):
        """
        Create SkyModel of the Diffuse Galactic sources.
        Maybe a repeat of code from the _cutout function.
        """
        template_diffuse = TemplateSpatialModel(diff_gal, normalize=False)
        source = SkyModel(
            spectral_model=PowerLawNormSpectralModel(),
            spatial_model=template_diffuse,
            name="diffuse-iem",
        )
        source.parameters["norm"].min = 0
        source.parameters["norm"].max = 10
        source.parameters["norm"].frozen = False

        return source

    def create_source_skymodel(self, src, aux_path, lp_is_intrinsic=False):
        """
        Build SkyModels from a given list of LAT files
        """
        source_name = src["@name"]
        spectrum_type = src["spectrum"]["@type"].split("EblAtten::")[-1]
        spectrum = src["spectrum"]["parameter"]
        spatial_pars = src["spatialModel"]["parameter"]

        source_name_red = source_name.replace("_", "").replace(" ", "")
        target_red = self.config_target.source_name.replace("_", "").replace(" ", "")

        # Check if target_source file exists
        if source_name_red == target_red:
            source_name = self.config_target.source_name
            is_src_target = True
            # self.log.debug("Detected target source")
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

                spec_model = SPECTRAL_MODEL_REGISTRY.get_cls(spectrum_type_final)()
                ebl_atten_pl = False

                if spectrum_type == "LogParabola" and "EblAtten" in src["spectrum"]["@type"]:
                    if lp_is_intrinsic:
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

        ebl_absorption_included = self.config_target.components.spectral.ebl_abs is not None

        if is_src_target and ebl_absorption_included:
            ebl_absorption = self.config_target.components.spectral.ebl_abs
            ebl_model = ebl_absorption.model_name
            ebl_spectral_model = EBLAbsorptionNormSpectralModel.read_builtin(
                ebl_model, redshift=ebl_absorption.redshift
            )
            spec_model = spec_model * ebl_spectral_model

        if src["spatialModel"]["@type"] == "SkyDirFunction":
            fk5_frame = SkyCoord(
                f"{spatial_pars[0]['@value']} deg",
                f"{spatial_pars[1]['@value']} deg",
                frame="fk5",
            )
            gal_frame = fk5_frame.transform_to("galactic")
            spatial_model = PointSpatialModel.from_position(gal_frame)
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

        return source_sky_model, is_src_target

    def xml_to_gammapy_model_params(
        self, params, spectrum_type, is_target=False, keep_sign=False, lp_is_intrinsic=False
    ):
        """
        Make some basic conversions on the spectral model parameters
        """
        params_list = []
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
                if par["@name"].lower() in ["cutoff"]:
                    new_par["name"] = "lambda_"
                    new_par["value"] = 1.0 / new_par["value"]
                    new_par["min"] = 1.0 / new_par["min"]
                    new_par["max"] = 1.0 / new_par["max"]
                    new_par["unit"] = "MeV-1"

            # More modifications:
            if new_par["name"] in ["reference", "ebreak", "emin", "emax"]:
                new_par["unit"] = "MeV"
            if new_par["name"] == "index" and not keep_sign:
                # Other than EBL Attenuated Power Law
                new_par["value"] *= -1
                new_par["min"] *= -1
                new_par["max"] *= -1
            new_par["error"] = 0
            params_list.append(new_par)

        params_final = Parameters.from_dict(params_list)

        return params_final

    def _counts_map(self):
        """
        Generate the counts Map object and fill it with the events information.
        """
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
            # self.log.info("Creating empty/dummy exclusion region")
            pos = SkyCoord(0, 90, unit="deg")
            exclusion_region = CircleSkyRegion(pos, 0.00001 * u.deg)
            self.exclusion_mask = ~excluded_geom.region_mask([exclusion_region])
        else:
            # self.log.info("Creating exclusion region")
            self.exclusion_mask = ~excluded_geom.region_mask(self.exclusion_regions)

    def add_source_to_exclusion_region(self, source_pos=None, radius=0.1 * u.deg):
        """
        Add to an existing exclusion_regions list, the source region.
        """
        if source_pos is not None:
            exclusion_region = CircleSkyRegion(
                center=source_pos.galactic,
                radius=radius,  # Generalize frame or ask from user.
            )
        else:
            sky_pos = self.config_target.sky_position
            source_pos = SkyCoord(
                u.Quantity(sky_pos.lon), u.Quantity(sky_pos.lat), frame=sky_pos.frame
            )
            exclusion_region = CircleSkyRegion(
                center=source_pos.galactic,
                radius=radius,
            )
        self.exclusion_regions.append(exclusion_region)

    def generate_dataset(self, key_name):
        """
        Generate MapDataset for the given Instrument files.
        """
        try:
            mask_safe = self.exclusion_mask
        except Exception:
            mask_bool = np.zeros(self.counts_map.geom.data_shape).astype("bool")
            mask_safe = Map.from_geom(self.counts_map.geom, mask_bool)
            mask_safe.data = np.asarray(mask_safe.data == 0, dtype=bool)

        dataset = MapDataset(
            models=self.target_full_model,
            counts=self.counts_map,
            exposure=self.exposure_interp,
            psf=self.psf,
            edisp=EDispKernelMap.from_edisp_kernel(self.edisp_interp_kernel),
            mask_safe=mask_safe,
            name="Fermi-LAT_{}".format(key_name),
        )

        return dataset
