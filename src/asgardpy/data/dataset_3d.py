"""
Main classes to define 3D Dataset Config, 3D Dataset Analysis Step and
to generate 3D Datasets from given Instruments' DL3 data from the config.
"""

import logging

import numpy as np
from astropy import units as u
from astropy.io import fits
from gammapy.catalog import CATALOG_REGISTRY
from gammapy.data import GTI, EventList
from gammapy.datasets import Datasets, MapDataset
from gammapy.irf import EDispKernel, EDispKernelMap, PSFMap
from gammapy.makers import MapDatasetMaker
from gammapy.maps import Map, MapAxis
from gammapy.modeling.models import Models

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
    read_models_from_asgardpy_config,
)
from asgardpy.gammapy.read_models import (
    create_gal_diffuse_skymodel,
    read_fermi_xml_models_list,
    update_aux_info_from_fermi_xml,
)
from asgardpy.io.input_dl3 import DL3Files, InputDL3Config
from asgardpy.io.io_dl4 import DL4BaseConfig, DL4Files, get_reco_energy_bins
from asgardpy.version import __public_version__

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
    key: list = []
    observation: ObservationsConfig = ObservationsConfig()
    map_selection: list[MapSelectionEnum] = MapDatasetMaker.available_selection
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
    input_dl3: list[InputDL3Config] = [InputDL3Config()]
    input_dl4: bool = False
    dataset_info: Dataset3DInfoConfig = Dataset3DInfoConfig()
    dl4_dataset_info: DL4BaseConfig = DL4BaseConfig()


class Dataset3DConfig(BaseConfig):
    """Config section for a list of all 3D DL3 Datasets information."""

    type: ReductionTypeEnum = ReductionTypeEnum.cube
    instruments: list[Dataset3DBaseConfig] = [Dataset3DBaseConfig()]


# The main Analysis Step
class Datasets3DAnalysisStep(AnalysisStepBase):
    """
    From the given config information, prepare the full list of 3D datasets,
    iterating over all the Instruments' information by running the
    Dataset3DGeneration function.

    Also calculate the total number of reconstructed energy bins used and the
    number of linked model parameters to incorporate in the total number of
    free model parameters, for the final estimation of total number of degrees
    of freedom.
    """

    tag = "datasets-3d"

    def _run(self):
        instruments_list = self.config.dataset3d.instruments
        self.log.info("%d number of 3D Datasets given", len(instruments_list))

        datasets_3d_final = Datasets()
        models_final = Models()
        instrument_spectral_info = {"name": [], "spectral_energy_ranges": []}

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

                self.update_model_dataset_names(models, dataset, models_final)
                dataset_instrument.append(dataset)

            models_final, free_params = self.link_diffuse_models(models_final, free_params)
            energy_bin_edges = dl4_files.get_spectral_energies()
            instrument_spectral_info["spectral_energy_ranges"].append(energy_bin_edges)

            for data in dataset_instrument:
                data._meta.optional = {
                    "instrument": config_3d_dataset.name,
                }
                data._meta.creation.creator += f", Asgardpy {__public_version__}"
                en_bins = get_reco_energy_bins(data, en_bins)
                datasets_3d_final.append(data)

        instrument_spectral_info["free_params"] = free_params
        instrument_spectral_info["en_bins"] = en_bins

        return datasets_3d_final, models_final, instrument_spectral_info

    def link_diffuse_models(self, models_final, free_params):
        """
        Function to link the diffuse models if present and reduce the number of
        degrees of freedom.
        """
        if len(models_final) > 0:
            diffuse_models_names = []
            for model_name in models_final.names:
                if "diffuse-iso" in model_name:
                    diffuse_models_names.append(model_name)

            if len(diffuse_models_names) > 1:
                for model_name in diffuse_models_names[1:]:
                    models_final[diffuse_models_names[0]].spectral_model.model2 = models_final[
                        model_name
                    ].spectral_model.model2
                    free_params -= 1
        else:
            models_final = None

        return models_final, free_params

    def update_model_dataset_names(self, models, dataset, models_final):
        """
        Function assigning datasets_names and including them in the final
        model list.

        When no associated list of models are provided, look for a separate
        model for target and an entry of catalog to fill in.
        """
        if len(models) > 0:
            for model_ in models:
                model_.datasets_names = [dataset.name]

                if model_.name in models_final.names:
                    models_final[model_.name].datasets_names.append(dataset.name)
                else:
                    models_final.append(model_)

        return models_final


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
            "transit_map": None,
        }
        self.events = {"events": None, "event_fits": None, "gti": None, "counts_map": None}
        self.diffuse_models = {
            "gal_diffuse": None,
            "iso_diffuse": None,
            "key_name": None,
            "gal_diffuse_cutout": None,
        }
        self.list_source_models = []

        # For updating the main config file with target source position
        # information if necessary.
        self.config_full = config_full
        self.config_target = config_full.target

    def run(self, key_name, filled_skymodel):
        """
        Main function to run the creation of 3D dataset.
        """
        # First check for the given file list if they are readable or not.
        non_gadf_file_list = self.read_to_objects(key_name)

        exclusion_regions = []

        if self.config_3d_dataset.input_dl3[0].type == "gadf-dl3":
            dataset = self.generate_gadf_dataset(exclusion_regions, filled_skymodel)

        elif "lat" in self.config_3d_dataset.input_dl3[0].type:
            dataset = self.generate_fermi_lat_dataset(non_gadf_file_list, exclusion_regions, key_name)
        elif self.config_3d_dataset.input_dl3[0].type == "hawc":
            dataset = self.generate_hawc_dataset(non_gadf_file_list, exclusion_regions)

        # Option for reading HAWC data
        if len(self.list_source_models) != 0:
            # Apply the same exclusion mask to the list of source models as applied
            # to the Counts Map
            self.list_source_models = apply_selection_mask_to_models(
                self.list_source_models,
                target_source=self.config_target.source_name,
                selection_mask=self.exclusion_mask,
            )

        return dataset, self.list_source_models

    def read_to_objects(self, key_name):
        """
        For each key type of files, read the files to get the required
        Gammapy objects for further analyses.

        Read the first IO list for events, IRFs and XML files, and then
        get the Diffuse models files list.
        """
        non_gadf_file_list = {}

        for io_dict in self.config_3d_dataset.input_dl3:
            match io_dict.type:
                case "hawc":
                    non_gadf_file_list, self.irfs["transit_map"] = self.get_base_objects(
                        io_dict, key_name, non_gadf_file_list
                    )

                case "lat":
                    (
                        non_gadf_file_list,
                        [
                            self.irfs["exposure"],
                            self.irfs["psf"],
                            self.irfs["edisp"],
                        ],
                    ) = self.get_base_objects(io_dict, key_name, non_gadf_file_list)

                case "lat-aux":
                    if io_dict.glob_pattern["iso_diffuse"] == "":
                        io_dict.glob_pattern = update_aux_info_from_fermi_xml(
                            io_dict.glob_pattern, non_gadf_file_list["xml_file"], fetch_iso_diff=True
                        )
                    if io_dict.glob_pattern["gal_diffuse"] == "":
                        io_dict.glob_pattern = update_aux_info_from_fermi_xml(
                            io_dict.glob_pattern, non_gadf_file_list["xml_file"], fetch_gal_diff=True
                        )

                    (
                        non_gadf_file_list,
                        [
                            self.diffuse_models["gal_diffuse"],
                            self.diffuse_models["iso_diffuse"],
                            self.diffuse_models["key_name"],
                        ],
                    ) = self.get_base_objects(io_dict, key_name, non_gadf_file_list)

                    self.list_source_models, self.diffuse_models = read_fermi_xml_models_list(
                        self.list_source_models,
                        io_dict.input_dir,
                        non_gadf_file_list["xml_file"],
                        self.diffuse_models,
                        asgardpy_target_config=self.config_target,
                    )

        # Check if the source position needs to be updated from the list provided.
        self.update_source_pos_from_3d_dataset()

        return non_gadf_file_list

    def get_base_objects(self, dl3_dir_dict, key, non_gadf_file_list):
        """
        For a DL3 files type and tag of the 'mode of observations' or key
        (FRONT/00 and BACK/01 for Fermi-LAT in enrico/fermipy files),
        read the files to appropriate Object type for further analysis.

        If there are no distinct key types of files, the value is None.
        """
        dl3_info = DL3Files(dl3_dir_dict, log=self.log)
        object_list = []

        if "lat" in dl3_dir_dict.type.lower():
            non_gadf_file_list = dl3_info.prepare_lat_files(key, non_gadf_file_list)

            if dl3_dir_dict.type.lower() in ["lat"]:
                exposure = Map.read(non_gadf_file_list["expmap_file"])
                psf = PSFMap.read(non_gadf_file_list["psf_file"], format="gtpsf")
                drmap = fits.open(non_gadf_file_list["edrm_file"])
                object_list = [exposure, psf, drmap]

            if dl3_dir_dict.type.lower() in ["lat-aux"]:
                object_list = [non_gadf_file_list["gal_diff_file"], non_gadf_file_list["iso_diff_file"], key]

        if dl3_dir_dict.type.lower() in ["hawc"]:
            dl3_info.list_dl3_files()
            non_gadf_file_list = dl3_info.dl3_index_files
            transit_map = Map.read(dl3_info.transit[0])
            object_list = transit_map

        return non_gadf_file_list, object_list

    def update_source_pos_from_3d_dataset(self):
        """
        Introduce the source coordinates from the 3D dataset to be the standard
        value in the main config file, for further use.
        """
        if self.config_target.use_uniform_position:
            source_position_from_3d = None

            for source in self.list_source_models:
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
        """
        diffuse_cutout = self.diffuse_models["gal_diffuse_map"].cutout(
            self.events["counts_map"].geom.center_skydir, self.events["counts_map"].geom.width[0]
        )
        self.diffuse_models["gal_diffuse_cutout"], _ = create_gal_diffuse_skymodel(diffuse_cutout)

        # Update the model in self.list_source_models
        for k, model_ in enumerate(self.list_source_models):
            if model_.name in ["diffuse-iem"]:
                self.list_source_models[k] = self.diffuse_models["gal_diffuse_cutout"]

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

    # Main functions for compiling different DL4 dataset generating procedures
    def generate_gadf_dataset(self, exclusion_regions, filled_skymodel):
        """
        Separate function containing the procedures on creating a GADF DL4
        dataset.

        non_gadf_file_list is not required here?
        """
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
        # Move it to Target module?
        if len(self.list_source_models) == 0:
            if not filled_skymodel:
                # Read the SkyModel info from AsgardpyConfig.target section
                if len(self.config_target.components) > 0:
                    models_ = read_models_from_asgardpy_config(self.config_target)
                    self.list_source_models = models_

                # If a catalog information is provided, use it to build up the list of models
                # Check if a catalog data is given with selection radius
                if self.config_target.use_catalog.selection_radius != 0 * u.deg:
                    catalog = CATALOG_REGISTRY.get_cls(self.config_target.use_catalog.name)()

                    # One can also provide a separate file, but one has to add
                    # another config option for reading Catalog file paths.
                    sep = catalog.positions.separation(center_pos["center"].galactic)

                    for k, cat_ in enumerate(catalog):
                        if sep[k] < self.config_target.use_catalog.selection_radius:
                            self.list_source_models.append(cat_.sky_model())

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
        return dataset

    def generate_fermi_lat_dataset(self, non_gadf_file_list, exclusion_regions, key_name):
        """
        Separate function containing the procedures on creating a Fermi-LAT DL4
        dataset.
        """
        self.load_events(non_gadf_file_list["events_file"])

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

        return dataset

    def update_hawc_energy_axis(self, bkg_geom, geom_config):
        """ """
        energy_edges = []
        for en in bkg_geom.axes["energy"].edges.value:
            energy_edges.append(round(en, 2))
        geom_config.axes[0].axis_custom.edges = energy_edges

        return geom_config

    def generate_hawc_dataset(self, non_gadf_file_list, exclusion_regions):
        """ """
        datasets_list = Datasets()

        for bin_id in self.config_3d_dataset.dataset_info.observation.event_type:
            self.log.info(f"Preparing for fHit number {bin_id}")
            observations = get_filtered_observations(
                dl3_path=self.config_3d_dataset.input_dl3[0].input_dir,
                dl3_index_files=non_gadf_file_list,
                event_type=bin_id,
                obs_config=self.config_3d_dataset.dataset_info.observation,
                log=self.log,
            )
            center_pos = get_source_position(target_region=self.config_3d_dataset.dataset_info.on_region)

            self.config_3d_dataset.dataset_info.geom = self.update_hawc_energy_axis(
                observations[0].bkg.geom,
                self.config_3d_dataset.dataset_info.geom,
            )

            geom = generate_geom(
                tag="3d",
                geom_config=self.config_3d_dataset.dataset_info.geom,
                center_pos=center_pos,
            )

            dataset_reference = get_dataset_reference(
                tag="3d",
                geom=geom,
                geom_config=self.config_3d_dataset.dataset_info.geom,
                name=f"{self.config_3d_dataset.name}_nHit-{bin_id}",  # "fhit "?
            )

            dataset_maker = get_dataset_maker(
                tag="3d",
                dataset_config=self.config_3d_dataset.dataset_info,
            )

            safe_maker = get_safe_mask_maker(safe_config=self.config_3d_dataset.dataset_info.safe_mask)
            transit_number = self.irfs["transit_map"].get_by_coord(geom.center_skydir)

            for obs in observations:
                dataset = dataset_maker.run(dataset_reference, obs)
                dataset.exposure.meta["livetime"] = (
                    6 * u.hour
                )  ## Put by hand from Gammapy tutorial. Make a new entry in config?
                dataset = safe_maker.run(dataset)
                ## Problem with stack_reduce - the PSFmap has a missing exposure map (IRFMap.stack, l905)

                dataset.background.data *= transit_number
                dataset.exposure.data *= transit_number

                datasets_list.append(dataset)

        return datasets_list
