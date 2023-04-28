"""
Main classes to define 3D Dataset Config, 3D Dataset Analysis Step and
to generate 3D Datasets from given Instruments' DL3 data from the config.
"""

import logging
import re
from typing import List

import numpy as np
import xmltodict
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

# from gammapy.analysis import Analysis, AnalysisConfig - no support for DL3 with RAD_MAX
from gammapy.data import GTI, EventList
from gammapy.datasets import Datasets, MapDataset
from gammapy.irf import EDispKernel, EDispKernelMap, PSFMap
from gammapy.makers import MapDatasetMaker
from gammapy.maps import Map, MapAxis
from gammapy.modeling.models import (
    Models,
    PowerLawNormSpectralModel,
    SkyModel,
    TemplateSpatialModel,
)
from regions import CircleAnnulusSkyRegion, CircleSkyRegion

from asgardpy.base import (
    AnalysisStepBase,
    BackgroundConfig,
    BaseConfig,
    GeomConfig,
    MapAxesConfig,
    MapSelectionEnum,
    ReductionTypeEnum,
    SafeMaskConfig,
    SpatialCircleConfig,
    get_energy_axis,
)
from asgardpy.data.target import (
    apply_selection_mask_to_models,
    create_gal_diffuse_skymodel,
    create_iso_diffuse_skymodel,
    create_source_skymodel,
)
from asgardpy.io import DL3Files, InputConfig

__all__ = [
    "Datasets3DAnalysisStep",
    "Dataset3DBaseConfig",
    "Dataset3DConfig",
    "Dataset3DGeneration",
    "Dataset3DInfoConfig",
]

log = logging.getLogger(__name__)


# Defining various components of 3D Dataset Config section
class Dataset3DInfoConfig(BaseConfig):
    name: str = "dataset-name"
    key: List = []
    map_selection: List[MapSelectionEnum] = MapDatasetMaker.available_selection
    geom: GeomConfig = GeomConfig()
    background: BackgroundConfig = BackgroundConfig()
    safe_mask: SafeMaskConfig = SafeMaskConfig()
    on_region: SpatialCircleConfig = SpatialCircleConfig()
    containment_correction: bool = True
    spectral_energy_range: MapAxesConfig = MapAxesConfig()


class Dataset3DBaseConfig(BaseConfig):
    name: str = "Instrument-name"
    io: List[InputConfig] = [InputConfig()]
    dataset_info: Dataset3DInfoConfig = Dataset3DInfoConfig()


class Dataset3DConfig(BaseConfig):
    type: ReductionTypeEnum = ReductionTypeEnum.cube
    instruments: List[Dataset3DBaseConfig] = [Dataset3DBaseConfig()]


# The main Analysis Step
class Datasets3DAnalysisStep(AnalysisStepBase):
    """
    From the given config information, prepare the full list of 3D datasets,
    iterating over all the Instruments' information by running the
    Dataset3DGeneration function.
    """

    tag = "datasets-3d"

    def _run(self):
        instruments_list = self.config.dataset3d.instruments
        self.log.info(f"{len(instruments_list)} number of 3D Datasets given")

        datasets_3d_final = Datasets()
        models_final = Models()
        instrument_spectral_info = {"name": [], "spectral_energy_ranges": []}

        # Iterate over all instrument information given:
        for i in np.arange(len(instruments_list)):
            config_3d_dataset = instruments_list[i]
            instrument_spectral_info["name"].append(config_3d_dataset.name)

            key_names = config_3d_dataset.dataset_info.key
            if len(key_names) > 0:
                self.log.info(f"The different keys used: {key_names}")
            else:
                key_names = [None]
                self.log.info("No distinct keys used for the 3D dataset")

            # Extra Datasets object to differentiate between datasets obtained
            # from various "keys" of each instrument.
            dataset_instrument = Datasets()

            # Retrieving a single dataset for each instrument.
            for key in key_names:
                generate_3d_dataset = Dataset3DGeneration(self.log, config_3d_dataset, self.config)
                dataset, models = generate_3d_dataset.run(key)

                # Assigning datasets_names and including them in the final
                # model list
                for model_ in models:
                    if key:
                        model_.datasets_names = [f"{config_3d_dataset.name}_{key}"]
                    else:
                        model_.datasets_names = [f"{config_3d_dataset.name}"]

                    if model_.name in models_final.names:
                        models_final[model_.name].datasets_names.append(model_.datasets_names[0])
                    else:
                        models_final.append(model_)

                dataset_instrument.append(dataset)

            # Linking the spectral model of the diffuse model for each key
            diffuse_models_names = []
            for model_name in models_final.names:
                if "diffuse-iso" in model_name:
                    diffuse_models_names.append(model_name)

            if len(diffuse_models_names) > 1:
                for model_name in diffuse_models_names[1:]:
                    models_final[diffuse_models_names[0]].spectral_model.model2 = models_final[
                        model_name
                    ].spectral_model.model2

            # Get the spectral energy information for each Instrument Dataset
            energy_axes = config_3d_dataset.dataset_info.spectral_energy_range
            if len(energy_axes.axis_custom.edges) > 0:
                energy_bin_edges = get_energy_axis(energy_axes, only_edges=True, custom_range=True)
            else:
                energy_bin_edges = get_energy_axis(
                    energy_axes,
                    only_edges=True,
                )

            instrument_spectral_info["spectral_energy_ranges"].append(energy_bin_edges)

            for data in dataset_instrument:
                datasets_3d_final.append(data)

        return datasets_3d_final, models_final, instrument_spectral_info


class Dataset3DGeneration:
    """
    Class for 3D dataset creation based on the config or AsgardpyConfig
    information provided on the 3D dataset and the target source.

    Runs the following steps:

    1. Read the DL3 files of 3D datasets into gammapy readable objects.

    2. Create the base counts Map.

    3. Prepare standard data reduction makers using the parameters passed in the config.

    4. Generate the final dataset.
    """

    def __init__(self, log, config_3d_dataset, config_full):
        self.config_3d_dataset = config_3d_dataset
        self.log = log
        self.exclusion_mask = None
        self.irfs = {
            "exposure": None,
            "psf": None,
            "edisp": None,
            "edisp_kernel": None,
            "edisp_interp_kernel": None,
            "exposure_interp": None,
        }
        self.events = {"events": None, "event_fits": None, "gti": None, "counts_map": None}
        self.diffuse_models = {"gal_diffuse": None, "iso_diffuse": None, "gal_diffuse_cutout": None}
        self.list_sources = []

        # For updating the main config file with target source position
        # information if necessary.
        self.config_full = config_full
        self.config_target = config_full.target

    def run(self, key_name):
        # First check for the given file list if they are readable or not.
        file_list = self.read_to_objects(key_name)

        # Start preparing objects to create the counts map
        self.set_energy_dispersion_matrix()
        self.load_events(file_list["events_file"])

        source_pos, evts_radius = self.get_source_skycoord()

        # Create the Counts Map
        self.create_counts_map(source_pos, evts_radius)

        # Create any dataset reduction makers or mask
        self.generate_diffuse_background_cutout()

        self.set_edisp_interpolator()
        self.set_exposure_interpolator()

        self.create_exclusion_mask()

        # Apply the same exclusion mask to the list of source models as applied
        # to the Counts Map
        self.list_sources = apply_selection_mask_to_models(
            self.list_sources,
            target_source=self.config_target.source_name,
            selection_mask=self.exclusion_mask,
        )

        # Generate the final dataset
        dataset = self.generate_dataset(key_name)

        return dataset, self.list_sources

    def read_to_objects(self, key_name):
        """
        For each key type of files, read the files to get the required
        Gammapy objects for further analyses.
        """
        file_list = {}

        # Read the first IO list for events, IRFs and XML files

        # Get the Diffuse models files list
        for io_ in self.config_3d_dataset.io:
            if io_.type in ["lat"]:
                file_list, [
                    self.irfs["exposure"],
                    self.irfs["psf"],
                    self.irfs["edisp"],
                ] = self.get_base_objects(io_, key_name, file_list)
                self.update_source_pos_from_3d_dataset()

            if io_.type in ["lat-aux"]:
                if io_.glob_pattern["iso_diffuse"] == "":
                    io_ = self.update_aux_info_from_xml(
                        io_, file_list["xml_file"], fetch_iso_diff=True
                    )
                if io_.glob_pattern["gal_diffuse"] == "":
                    io_ = self.update_aux_info_from_xml(
                        io_, file_list["xml_file"], fetch_gal_diff=True
                    )

                file_list, [
                    self.diffuse_models["gal_diffuse"],
                    self.diffuse_models["iso_diffuse"],
                ] = self.get_base_objects(io_, key_name, file_list)
                self.get_list_objects(io_.input_dir, file_list["xml_file"])

        return file_list

    def update_aux_info_from_xml(
        self, io_dict, xml_file, fetch_iso_diff=False, fetch_gal_diff=False
    ):
        """
        When no glob_search patterns on axuillary files are provided, fetch
        them from the XML file and update the AsgardpyConfig object.

        Currently assuming this to be applicable only for Fermi-LAT data.
        """
        with open(xml_file) as file:
            data = xmltodict.parse(file.read())["source_library"]["source"]

        for source in data:
            source_name = source["@name"]
            if source_name in ["IsoDiffModel", "isodiff"]:
                if fetch_iso_diff:
                    file_path = source["spectrum"]["@file"]
                    file_name = file_path.split("/")[-1]
                    io_dict.glob_pattern["iso_diffuse"] = file_name
            if source_name in ["GalDiffModel", "galdiff"]:
                if fetch_gal_diff:
                    file_path = source["spatialModel"]["@file"]
                    file_name = file_path.split("/")[-1]
                    io_dict.glob_pattern["gal_diffuse"] = file_name
        return io_dict

    def get_base_objects(self, dl3_dir_dict, key, file_list):
        """
        For a DL3 files type and tag of the 'mode of observations' or key
        (FRONT/00 and BACK/01 for Fermi-LAT in enrico/fermipy files),
        read the files to appropriate Object type for further analysis.

        If there are no distinct key types of files, the value is None.
        """
        dl3_info = DL3Files(dl3_dir_dict, file_list, log=self.log)
        file_list = dl3_info.prepare_lat_files(key, file_list)
        object_list = []

        if dl3_dir_dict.type.lower() in ["lat"]:
            exposure = Map.read(file_list["expmap_file"])
            psf = PSFMap.read(file_list["psf_file"], format="gtpsf")
            drmap = fits.open(file_list["edrm_file"])
            object_list = [exposure, psf, drmap]

        if dl3_dir_dict.type.lower() in ["lat-aux"]:
            diff_gal = Map.read(file_list["gal_diff_file"])
            diff_gal.meta["filename"] = file_list["gal_diff_file"]
            diff_iso = create_iso_diffuse_skymodel(file_list["iso_diff_file"], key)
            object_list = [diff_gal, diff_iso]

        return file_list, object_list

    def get_list_objects(self, aux_path, xml_file):
        """
        Read from the XML file to enlist the various objects and get their
        SkyModels

        Currently assuming this to be applicable only for Fermi-LAT data.
        """
        with open(xml_file) as file:
            data = xmltodict.parse(file.read())["source_library"]["source"]

        is_target_source = False
        for source in data:
            source_name = source["@name"]
            if source_name in ["IsoDiffModel", "isodiff"]:
                source = self.diffuse_models["iso_diffuse"]
            elif source_name in ["GalDiffModel", "galdiff"]:
                source = create_gal_diffuse_skymodel(self.diffuse_models["gal_diffuse"])
            else:
                source, is_target_source = create_source_skymodel(
                    self.config_target, source, aux_path
                )
            if is_target_source:
                self.list_sources.insert(0, source)
                is_target_source = False  # To not let it repeat
            else:
                self.list_sources.append(source)

    def update_source_pos_from_3d_dataset(self):
        """
        Introduce the source coordinates from the 3D dataset to be the standard
        value in the main config file, for further use.
        """
        if self.config_target.use_uniform_position:
            source_position_from_3d = None

            for source in self.list_sources:
                if source.name == self.config_target.source_name:
                    source_position_from_3d = source.spatial_model.position.icrs

                    self.config_full.target.sky_position.lon = str(
                        u.Quantity(source_position_from_3d.ra)
                    )
                    self.config_full.target.sky_position.lat = str(
                        u.Quantity(source_position_from_3d.dec)
                    )

                    self.config_full.update(self.config_full)
                    self.config_target = self.config_full.target

    def set_energy_axes(self):
        """
        Get the energy axes from the given Detector Response Matrix file.

        Needs to be generalized for other possible file structures for other
        instruments.
        """
        energy_lo = self.irfs["edisp"]["DRM"].data["ENERG_LO"] * u.MeV
        energy_hi = self.irfs["edisp"]["DRM"].data["ENERG_HI"] * u.MeV

        energy_axis = MapAxis.from_energy_edges(np.append(energy_lo[0], energy_hi))
        energy_axis_true = energy_axis.copy(name="energy_true")

        return energy_axis, energy_axis_true

    def set_energy_dispersion_matrix(self):
        """
        Generate the Energy Dispersion Kernel from the given Detector Response
        Matrix file.

        Needs to be generalized for other possible file structures for other
        instruments.
        """
        energy_axis, energy_axis_true = self.set_energy_axes()
        drm = self.irfs["edisp"]["DRM"].data["MATRIX"]
        drm_matrix = np.array(list(drm))

        self.irfs["edisp_kernel"] = EDispKernel(
            axes=[energy_axis_true, energy_axis], data=drm_matrix
        )

    def load_events(self, events_file):
        """
        Loading the events files for the specific "Key" into an EventList
        object and the GTI information into a GTI object.
        """
        self.events["event_fits"] = fits.open(events_file)
        self.events["events"] = EventList.read(events_file)
        self.events["gti"] = GTI.read(events_file)

    def get_source_skycoord(self):
        """
        Get the source skycoord and the ROI radius from the events file.

        Needs to be generalized for other possible file structures for other
        instruments.
        """
        try:
            dsval2 = self.events["event_fits"][1].header["DSVAL2"]
            list_str_check = re.findall(r"[-+]?\d*\.\d+|\d+", dsval2)
            ra_pos, dec_pos, evts_radius = [float(k) for k in list_str_check]
        except IndexError:
            history = str(self.events["event_fits"][1].header["HISTORY"])
            str_ = history.split("angsep(RA,DEC,")[1]
            list_str_check = re.findall(r"[-+]?\d*\.\d+|\d+", str_)[:3]
            ra_pos, dec_pos, evts_radius = [float(k) for k in list_str_check]

        source_pos = SkyCoord(ra_pos, dec_pos, unit="deg", frame="fk5")

        return source_pos, evts_radius

    def create_counts_map(self, source_pos, evts_radius):
        """
        Generate the counts Map object using the information provided in the
        geom section of the Config and fill it with the events' RA-Dec
        position, Energy and Time information.
        """
        geom_config = self.config_3d_dataset.dataset_info.geom

        energy_axes = geom_config.axes[0]
        energy_axis = get_energy_axis(energy_axes)
        bin_size = geom_config.wcs.binsize.to_value(u.deg)

        if geom_config.from_events_file:
            self.events["counts_map"] = Map.create(
                skydir=source_pos.galactic,
                binsz=bin_size,
                npix=(
                    int(evts_radius * 2 / bin_size),
                    int(evts_radius * 2 / bin_size),
                ),  # Using the limits from the events fits file
                proj=geom_config.wcs.proj,
                frame="galactic",
                axes=[energy_axis],
                dtype=float,
            )
        else:
            width_ = geom_config.wcs.map_frame_shape.width.to_value(u.deg)
            width_in_pixel = int(width_ / bin_size)
            height_ = geom_config.wcs.map_frame_shape.height.to_value(u.deg)
            height_in_pixel = int(height_ / bin_size)

            self.events["counts_map"] = Map.create(
                skydir=source_pos.galactic,
                binsz=bin_size,
                npix=(width_in_pixel, height_in_pixel),
                proj=geom_config.wcs.proj,
                frame="galactic",
                axes=[energy_axis],
                dtype=float,
            )
        self.events["counts_map"].fill_by_coord(
            {
                "skycoord": self.events["events"].radec,
                "energy": self.events["events"].energy,
                "time": self.events["events"].time,
            }
        )

    def generate_diffuse_background_cutout(self):
        """
        Perform a cutout of the Diffuse background model with respect to the
        counts map geom (may improve fitting speed?) and update the main list
        of models.

        The Template Spatial Model is without normalization currently.
        """
        diffuse_cutout = self.diffuse_models["gal_diffuse"].cutout(
            self.events["counts_map"].geom.center_skydir, self.events["counts_map"].geom.width[0]
        )

        template_diffuse = TemplateSpatialModel(diffuse_cutout, normalize=False)

        self.diffuse_models["gal_diffuse_cutout"] = SkyModel(
            spectral_model=PowerLawNormSpectralModel(),
            spatial_model=template_diffuse,
            name="diffuse-iem",
        )
        self.diffuse_models["gal_diffuse_cutout"].parameters["norm"].min = 0
        self.diffuse_models["gal_diffuse_cutout"].parameters["norm"].max = 10
        self.diffuse_models["gal_diffuse_cutout"].parameters["norm"].frozen = False

        # Update the model in self.list_sources
        for k, model_ in enumerate(self.list_sources):
            if model_.name in ["diffuse-iem"]:
                self.list_sources[k] = self.diffuse_models["gal_diffuse_cutout"]

    def set_edisp_interpolator(self):
        """
        Get the Energy Dispersion Kernel interpolated along true and
        reconstructed energy of the real counts.
        """
        axis_reco = MapAxis.from_edges(
            self.events["counts_map"].geom.axes["energy"].edges,
            name="energy",
            unit="MeV",  # Need to be generalized
            interp="log",
        )
        axis_true = axis_reco.copy(name="energy_true")
        energy_reco, energy_true = np.meshgrid(axis_true.center, axis_reco.center)

        drm_interp = self.irfs["edisp_kernel"].evaluate(
            "linear", **{"energy": energy_reco, "energy_true": energy_true}
        )
        self.irfs["edisp_interp_kernel"] = EDispKernel(
            axes=[axis_true, axis_reco], data=np.asarray(drm_interp)
        )

    def set_exposure_interpolator(self):
        """
        Set Exposure interpolated along energy axis of real counts.
        """
        self.irfs["exposure_interp"] = self.irfs["exposure"].interp_to_geom(
            self.events["counts_map"].geom.as_energy_true
        )

    def create_exclusion_mask(self):
        """
        Generate an Exclusion Mask for the final MapDataset and also select
        the region for the list of Models using the excluded regions.
        """
        exclusion_regions = []
        excluded_geom = self.events["counts_map"].geom.copy()
        exclusion_params = self.config_3d_dataset.dataset_info.background.exclusion

        if len(exclusion_params.regions) != 0:
            for region in exclusion_params.regions:
                if region.name == "None":
                    coord = region.position
                    center_ex = SkyCoord(
                        u.Quantity(coord.lon), u.Quantity(coord.lat), frame=coord.frame
                    ).icrs
                else:
                    center_ex = SkyCoord.from_name(region.name)

                if region.type == "CircleAnnulusSkyRegion":
                    excluded_region = CircleAnnulusSkyRegion(
                        center=center_ex,
                        inner_radius=u.Quantity(region.parameters["rad_0"]),
                        outer_radius=u.Quantity(region.parameters["rad_1"]),
                    )
                elif region.type == "CircleSkyRegion":
                    excluded_region = CircleSkyRegion(
                        center=center_ex, radius=u.Quantity(region.parameters["region_radius"])
                    )
                else:
                    self.log.error(f"Unknown type of region passed {region.type}")
                exclusion_regions.append(excluded_region)
            self.exclusion_mask = ~excluded_geom.region_mask(exclusion_regions)

        elif len(exclusion_regions) == 0:
            self.log.info("Creating empty/dummy exclusion region")
            pos = SkyCoord(0, 90, unit="deg")
            exclusion_region = CircleSkyRegion(pos, 0.00001 * u.deg)

            exclusion_regions.append(exclusion_region)
            self.exclusion_mask = ~excluded_geom.region_mask(exclusion_regions)
        else:
            self.log.info("Creating exclusion region")
            self.exclusion_mask = ~excluded_geom.region_mask(exclusion_regions)

    def add_source_to_exclusion_region(
        self, exclusion_regions, source_pos=None, radius=0.1 * u.deg
    ):
        """
        Add to an existing exclusion_regions list, the source region.

        Check if this function is necessary.
        """
        if source_pos is not None:
            exclusion_region = CircleSkyRegion(
                center=source_pos.galactic,
                radius=radius,
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
        exclusion_regions.append(exclusion_region)

        return exclusion_regions

    def generate_dataset(self, key_name):
        """
        Generate MapDataset for the given Instrument files using the Counts Map,
        and IRFs objects.
        """
        try:
            mask_safe = self.exclusion_mask
            self.log.info("Using the exclusion mask to create a safe mask")
        except ValueError:
            self.log.info("Using counts_map to create safe mask")
            mask_bool = np.zeros(self.events["counts_map"].geom.data_shape).astype("bool")
            mask_safe = Map.from_geom(self.events["counts_map"].geom, mask_bool)
            mask_safe.data = np.asarray(mask_safe.data == 0, dtype=bool)

        edisp = EDispKernelMap.from_edisp_kernel(self.irfs["edisp_interp_kernel"])
        if key_name:
            name = f"{self.config_3d_dataset.name}_{key_name}"
        else:
            name = f"{self.config_3d_dataset.name}"

        dataset = MapDataset(
            counts=self.events["counts_map"],
            gti=self.events["gti"],
            exposure=self.irfs["exposure_interp"],
            psf=self.irfs["psf"],
            edisp=edisp,
            mask_safe=mask_safe,
            name=name,
        )

        return dataset
