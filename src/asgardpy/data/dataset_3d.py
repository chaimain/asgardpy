"""
Main classes to define 3D Dataset Config, 3D Dataset Analysis Step and
to generate 3D Datasets from given Instruments' DL3 data from the config.
"""

import logging
from typing import List

import numpy as np
import xmltodict
from astropy import units as u
from astropy.io import fits

# from gammapy.analysis import Analysis, AnalysisConfig - no support for DL3 with RAD_MAX
from gammapy.catalog import CATALOG_REGISTRY
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

from asgardpy.analysis.step_base import AnalysisStepBase
from asgardpy.base.base import BaseConfig
from asgardpy.base.geom import (
    GeomConfig,
    SkyPositionConfig,
    create_counts_map,
    generate_geom,
    get_source_position,
)
from asgardpy.base.reduction import (
    BackgroundConfig,
    MapSelectionEnum,
    ObservationsConfig,
    ReductionTypeEnum,
    SafeMaskConfig,
    generate_dl4_dataset,
    get_bkg_maker,
    get_dataset_maker,
    get_dataset_reference,
    get_exclusion_region_mask,
    get_filtered_observations,
    get_safe_mask_maker,
)
from asgardpy.data.target import (
    apply_selection_mask_to_models,
    create_gal_diffuse_skymodel,
    create_iso_diffuse_skymodel,
    create_source_skymodel,
    read_models_from_asgardpy_config,
)
from asgardpy.io.input_dl3 import DL3Files, InputDL3Config
from asgardpy.io.io_dl4 import DL4BaseConfig, DL4Files, get_reco_energy_bins

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
    """Config section for 3D DL3 Dataset Reduction for each instrument."""

    name: str = "dataset-name"
    key: List = []
    observation: ObservationsConfig = ObservationsConfig()
    map_selection: List[MapSelectionEnum] = MapDatasetMaker.available_selection
    geom: GeomConfig = GeomConfig()
    background: BackgroundConfig = BackgroundConfig()
    safe_mask: SafeMaskConfig = SafeMaskConfig()
    on_region: SkyPositionConfig = SkyPositionConfig()
    containment_correction: bool = True


class Dataset3DBaseConfig(BaseConfig):
    """
    Config section for 3D DL3 Dataset base information for each instrument.
    """

    name: str = "Instrument-name"
    input_dl3: List[InputDL3Config] = [InputDL3Config()]
    input_dl4: bool = False
    dataset_info: Dataset3DInfoConfig = Dataset3DInfoConfig()
    dl4_dataset_info: DL4BaseConfig = DL4BaseConfig()


class Dataset3DConfig(BaseConfig):
    """Config section for a list of all 3D DL3 Datasets information."""

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
        self.log.info("%d number of 3D Datasets given", len(instruments_list))

        datasets_3d_final = Datasets()
        models_final = Models()
        instrument_spectral_info = {"name": [], "spectral_energy_ranges": []}

        # Calculate the total number of reconstructed energy bins used
        # and the number of linked model parameters to incorporate in the
        # total number of free model parameters, for the final estimation of
        # total number of degrees of freedom
        free_params = 0
        en_bins = 0

        # Iterate over all instrument information given:
        for i in np.arange(len(instruments_list)):
            config_3d_dataset = instruments_list[i]
            instrument_spectral_info["name"].append(config_3d_dataset.name)

            key_names = config_3d_dataset.dataset_info.key
            if len(key_names) > 0:
                keys_str = " ".join(map(str, key_names))
                self.log.info("The different keys used: %s", keys_str)
            else:
                key_names = [None]
                self.log.info("No distinct keys used for the 3D dataset")

            # Extra Datasets object to differentiate between datasets obtained
            # from various "keys" of each instrument.
            dataset_instrument = Datasets()
            dl4_files = DL4Files(config_3d_dataset.dl4_dataset_info, self.log)

            # Only read unique SkyModels for the first instrument, unless there
            # are associated files like XML to read from for the particular instrument.
            filled_skymodel = False
            if len(models_final) > 0:
                filled_skymodel = True

            # Retrieving a single dataset for each instrument.
            for key in key_names:
                if not config_3d_dataset.input_dl4:
                    generate_3d_dataset = Dataset3DGeneration(self.log, config_3d_dataset, self.config)
                    dataset, models = generate_3d_dataset.run(key, filled_skymodel)
                else:
                    dataset = dl4_files.get_dl4_dataset(config_3d_dataset.dataset_info.observation)
                    models = []

                # Use the individual Dataset type object for following tasks
                if isinstance(dataset, Datasets):
                    dataset = dataset[0]

                # Assigning datasets_names and including them in the final
                # model list

                # When no associated list of models are provided, look for a
                # separate model for target and an entry of catalog to fill in.
                if len(models) > 0:
                    for model_ in models:
                        model_.datasets_names = [dataset.name]

                        if model_.name in models_final.names:
                            models_final[model_.name].datasets_names.append(dataset.name)
                        else:
                            models_final.append(model_)

                dataset_instrument.append(dataset)

            if len(models_final) > 0:
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
                        # For each linked model parameter, reduce the number of DoF
                        free_params -= 1
            else:
                models_final = None

            energy_bin_edges = dl4_files.get_spectral_energies()
            instrument_spectral_info["spectral_energy_ranges"].append(energy_bin_edges)

            for data in dataset_instrument:
                en_bins = get_reco_energy_bins(data, en_bins)
                datasets_3d_final.append(data)

        instrument_spectral_info["free_params"] = free_params
        instrument_spectral_info["en_bins"] = en_bins

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

    def run(self, key_name, filled_skymodel):
        """
        Main function to run the creation of 3D dataset.
        """
        # First check for the given file list if they are readable or not.
        file_list = self.read_to_objects(key_name)
        # self.log.info(file_list)

        exclusion_regions = []

        if self.config_3d_dataset.input_dl3[0].type == "gadf-dl3":
            observations = get_filtered_observations(
                dl3_path=self.config_3d_dataset.input_dl3[0].input_dir,
                obs_config=self.config_3d_dataset.dataset_info.observation,
                log=self.log,
            )
            center_pos = get_source_position(target_region=self.config_3d_dataset.dataset_info.on_region)

            geom = generate_geom(
                tag="3d",
                geom_config=self.config_3d_dataset.dataset_info.geom,
                center_pos=center_pos,
            )

            dataset_reference = get_dataset_reference(
                tag="3d", geom=geom, geom_config=self.config_3d_dataset.dataset_info.geom
            )

            dataset_maker = get_dataset_maker(
                tag="3d",
                dataset_config=self.config_3d_dataset.dataset_info,
            )

            safe_maker = get_safe_mask_maker(safe_config=self.config_3d_dataset.dataset_info.safe_mask)

            # If there is no explicit list of models provided for the 3D data,
            # one can use one of the several catalogs available in Gammapy.
            # Reading them as Models will keep the procedure uniform.

            # Unless the unique skymodels for 3D dataset is already set.

            if len(self.list_sources) == 0:
                if not filled_skymodel:
                    # Read the SkyModel info from AsgardpyConfig.target section
                    if len(self.config_target.components) > 0:
                        models_ = read_models_from_asgardpy_config(self.config_target)
                        self.list_sources = models_

                    # If a catalog information is provided, use it to build up the list of models
                    # Check if a catalog data is given with selection radius
                    if self.config_target.use_catalog.selection_radius != 0 * u.deg:
                        catalog = CATALOG_REGISTRY.get_cls(self.config_target.use_catalog.name)()

                        # One can also provide a separate file, but one has to add
                        # another config option for reading Catalog file paths.
                        sep = catalog.positions.separation(center_pos["center"].galactic)

                        # base_geom = geom.copy()
                        # inside_geom = base_geom.to_image().contains(catalog.positions)

                        # idx_list = np.nonzero(inside_geom)[0]
                        # for i in idx_list:
                        for k, cat_ in enumerate(catalog):
                            if sep[k] < self.config_target.use_catalog.selection_radius:
                                self.list_sources.append(cat_.sky_model())

            excluded_geom = generate_geom(
                tag="3d-ex",
                geom_config=self.config_3d_dataset.dataset_info.geom,
                center_pos=center_pos,
            )

            exclusion_mask = get_exclusion_region_mask(
                exclusion_params=self.config_3d_dataset.dataset_info.background.exclusion,
                exclusion_regions=exclusion_regions,
                excluded_geom=excluded_geom,
                config_target=self.config_target,
                geom_config=self.config_3d_dataset.dataset_info.geom,
                log=self.log,
            )

            bkg_maker = get_bkg_maker(
                bkg_config=self.config_3d_dataset.dataset_info.background,
                exclusion_mask=exclusion_mask,
            )

            dataset = generate_dl4_dataset(
                tag="3d",
                observations=observations,
                dataset_reference=dataset_reference,
                dataset_maker=dataset_maker,
                bkg_maker=bkg_maker,
                safe_maker=safe_maker,
                n_jobs=self.config_full.general.n_jobs,
                parallel_backend=self.config_full.general.parallel_backend,
            )

        elif "lat" in self.config_3d_dataset.input_dl3[0].type:
            self.load_events(file_list["events_file"])

            # Start preparing objects to create the counts map
            self.set_energy_dispersion_matrix()

            center_pos = get_source_position(
                target_region=self.config_target.sky_position,
                fits_header=self.events["event_fits"][1].header,
            )

            # Create the Counts Map
            self.events["counts_map"] = create_counts_map(
                geom_config=self.config_3d_dataset.dataset_info.geom,
                center_pos=center_pos,
            )
            self.events["counts_map"].fill_by_coord(
                {
                    "skycoord": self.events["events"].radec,
                    "energy": self.events["events"].energy,
                    "time": self.events["events"].time,
                }
            )
            # Create any dataset reduction makers or mask
            self.generate_diffuse_background_cutout()

            self.set_edisp_interpolator()
            self.set_exposure_interpolator()

            self.exclusion_mask = get_exclusion_region_mask(
                exclusion_params=self.config_3d_dataset.dataset_info.background.exclusion,
                excluded_geom=self.events["counts_map"].geom.copy(),
                exclusion_regions=exclusion_regions,
                config_target=self.config_target,
                geom_config=self.config_3d_dataset.dataset_info.geom,
                log=self.log,
            )

            # Generate the final dataset
            dataset = self.generate_dataset(key_name)

        if len(self.list_sources) != 0:
            # Apply the same exclusion mask to the list of source models as applied
            # to the Counts Map
            self.list_sources = apply_selection_mask_to_models(
                self.list_sources,
                target_source=self.config_target.source_name,
                selection_mask=self.exclusion_mask,
            )

        return dataset, self.list_sources

    def read_to_objects(self, key_name):
        """
        For each key type of files, read the files to get the required
        Gammapy objects for further analyses.
        """
        file_list = {}

        # Read the first IO list for events, IRFs and XML files

        # Get the Diffuse models files list
        for io_ in self.config_3d_dataset.input_dl3:
            if io_.type in ["gadf-dl3"]:
                file_list, _ = self.get_base_objects(io_, key_name, file_list)

            if io_.type in ["lat"]:
                file_list, [
                    self.irfs["exposure"],
                    self.irfs["psf"],
                    self.irfs["edisp"],
                ] = self.get_base_objects(io_, key_name, file_list)

            if io_.type in ["lat-aux"]:
                if io_.glob_pattern["iso_diffuse"] == "":
                    io_ = self.update_aux_info_from_xml(io_, file_list["xml_file"], fetch_iso_diff=True)
                if io_.glob_pattern["gal_diffuse"] == "":
                    io_ = self.update_aux_info_from_xml(io_, file_list["xml_file"], fetch_gal_diff=True)

                file_list, [
                    self.diffuse_models["gal_diffuse"],
                    self.diffuse_models["iso_diffuse"],
                ] = self.get_base_objects(io_, key_name, file_list)
                self.get_list_objects(io_.input_dir, file_list["xml_file"])

        # After reading the list of source objects, check if the source position needs to be
        # updated from the list provided.
        self.update_source_pos_from_3d_dataset()

        return file_list

    def update_aux_info_from_xml(self, io_dict, xml_file, fetch_iso_diff=False, fetch_gal_diff=False):
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
        dl3_info = DL3Files(dl3_dir_dict, log=self.log)
        object_list = []

        if dl3_dir_dict.type.lower() in ["gadf-dl3"]:
            dl3_info.list_dl3_files()
            file_list = dl3_info.events_files

            return file_list, object_list
        else:
            file_list = dl3_info.prepare_lat_files(key, file_list)

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
        with open(xml_file, encoding="utf-8") as file:
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
                    self.config_target,
                    source,
                    aux_path,
                    base_model_type="Fermi-XML",
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

                    self.config_full.target.sky_position.lon = str(u.Quantity(source_position_from_3d.ra))
                    self.config_full.target.sky_position.lat = str(u.Quantity(source_position_from_3d.dec))

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

        self.irfs["edisp_kernel"] = EDispKernel(axes=[energy_axis_true, energy_axis], data=drm_matrix)

    def load_events(self, events_file):
        """
        Loading the events files for the specific "Key" into an EventList
        object and the GTI information into a GTI object.
        """
        self.events["event_fits"] = fits.open(events_file)
        self.events["events"] = EventList.read(events_file)
        self.events["gti"] = GTI.read(events_file)

    def generate_diffuse_background_cutout(self):
        """
        Perform a cutout of the Diffuse background model with respect to the
        counts map geom (may improve fitting speed?) and update the main list
        of models.

        The reference Spatial Model is without normalization currently.
        """
        diffuse_cutout = self.diffuse_models["gal_diffuse"].cutout(
            self.events["counts_map"].geom.center_skydir, self.events["counts_map"].geom.width[0]
        )

        reference_diffuse = TemplateSpatialModel(diffuse_cutout, normalize=False)

        self.diffuse_models["gal_diffuse_cutout"] = SkyModel(
            spectral_model=PowerLawNormSpectralModel(),
            spatial_model=reference_diffuse,
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
        self.irfs["edisp_interp_kernel"] = EDispKernel(axes=[axis_true, axis_reco], data=np.asarray(drm_interp))

    def set_exposure_interpolator(self):
        """
        Set Exposure interpolated along energy axis of real counts.
        """
        self.irfs["exposure_interp"] = self.irfs["exposure"].interp_to_geom(
            self.events["counts_map"].geom.as_energy_true
        )

    def generate_dataset(self, key_name):
        """
        Generate MapDataset for the given Instrument files using the Counts Map,
        and IRFs objects.
        """
        if self.exclusion_mask is not None:
            mask_safe = self.exclusion_mask
            self.log.info("Using the exclusion mask to create a safe mask")
        else:
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
