"""
Generating 3D Datasets from given Instrument DL3 data
"""

import logging
from typing import List

import numpy as np
import xmltodict
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

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

from asgardpy.data.base import AnalysisStepBase, BaseConfig, TimeIntervalsConfig
from asgardpy.data.geom import EnergyAxisConfig, SpatialCircleConfig
from asgardpy.data.reduction import (
    BackgroundConfig,
    MapSelectionEnum,
    ReductionTypeEnum,
    SafeMaskConfig,
)
from asgardpy.data.target import (
    create_gal_diffuse_skymodel,
    create_iso_diffuse_skymodel,
    create_source_skymodel,
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
    obs_time: TimeIntervalsConfig = TimeIntervalsConfig()
    background: BackgroundConfig = BackgroundConfig()
    safe_mask: SafeMaskConfig = SafeMaskConfig()
    on_region: SpatialCircleConfig = SpatialCircleConfig()
    containment_correction: bool = True
    spectral_energy_range: EnergyAxisConfig = EnergyAxisConfig()


class Dataset3DBaseConfig(BaseConfig):
    name: str = "Instrument-name"
    io: List[InputConfig] = [InputConfig()]
    dataset_info: Dataset3DInfoConfig = Dataset3DInfoConfig()


class Dataset3DConfig(BaseConfig):
    type: ReductionTypeEnum = ReductionTypeEnum.cube
    instruments: List[Dataset3DBaseConfig] = [Dataset3DBaseConfig()]


class Datasets3DAnalysisStep(AnalysisStepBase):
    """
    From the given config information, prepare the full list of 3D datasets.
    """

    tag = "datasets-3d"

    def _run(self):
        # Iterate over all instrument information given:
        instruments_list = self.config.dataset3d.instruments
        self.log.info(f"{len(instruments_list)} number of 3D Datasets given")

        datasets_3d_final = Datasets()
        models_final = Models()
        spectral_energy_ranges = []

        for i in np.arange(len(instruments_list)):
            self.config_3d_dataset = instruments_list[i]
            key_names = self.config_3d_dataset.dataset_info.key
            self.log.info(f"The different keys used: {key_names}")

            # Retrieving a single dataset for each instrument.
            dataset_instrument = Datasets()
            for key in key_names:
                generate_3d_dataset = Dataset3DGeneration(
                    self.log, self.config_3d_dataset, self.config, key
                )
                dataset, models, mask_exclusion = generate_3d_dataset.run()
                print(mask_exclusion)
                for m in models:
                    # Assigning datasets_names
                    #if m.name != self.config.target.source_name:
                    m.datasets_names = [f"{self.config_3d_dataset.name}_{key}"]
                    print(m.datasets_names)
                    if m.name in models_final.names:
                        #print(models_final[m.name].datasets_names, "before any change", type(models_final[m.name].datasets_names))
                        models_final[m.name].datasets_names.append(m.datasets_names[0])
                        #print(models_final[m.name].datasets_names, "after any change", type(models_final[m.name].datasets_names))
                    else:
                        #print(m, "is new model to be added")
                        models_final.append(m)
                #print(models_final)

                dataset_instrument.append(dataset)
            diffuse_models_names = []
            for m in models_final.names:
                if "diffuse-iso" in m:
                    diffuse_models_names.append(m)
            if len(diffuse_models_names)>1:
                for n in diffuse_models_names[1:]:
                    #print(models_final[diffuse_models_names[0]].spectral_model.model2, models_final[n].spectral_model.model2)
                    models_final[diffuse_models_names[0]].spectral_model.model2 = models_final[n].spectral_model.model2
            #print(models_final)


            # Get the spectral energy information for each Instrument Dataset
            energy_range = self.config_3d_dataset.dataset_info.spectral_energy_range
            energy_bin_edges = MapAxis.from_energy_bounds(
                energy_min=u.Quantity(energy_range.min),
                energy_max=u.Quantity(energy_range.max),
                nbin=int(energy_range.nbins),
                per_decade=True,
            ).edges

            if self.config.general.stacked_dataset:
                # Add a condition on appending names of models for different keys,
                # except when it is key specific like the diffuse iso models
                """
                for d in dataset_instrument:
                    for m in d.models:
                        print(m.name)
                        if "diffuse-iso" in m.name:
                            print(f"Got the special diffuse Iso model with name {m.name}")
                            print(m.datasets_names)
                            #print("The second component of the spectral model is", m.spectral_model.model2)
                            # Trying to keep the 2 diffuse models not be stacked together under the same datasets name, by keeping is a list
                            #m.datasets_names = [m.datasets_names]
                        else:
                            #m.datasets_names = [f"{self.config_3d_dataset.name}_{key}" for key in key_names]
                            print(m.datasets_names)
                """
                dataset_instrument.stack_reduce(name=self.config_3d_dataset.name)
                print(dataset_instrument)

                for data in dataset_instrument:
                    # Check each models' datasets names to confirm
                    #print(data.models)
                    datasets_3d_final.append(data)
                #datasets_3d_final.append(data)
                spectral_energy_ranges.append(energy_bin_edges)
            else:
                for data in dataset_instrument:
                    datasets_3d_final.append(data)
                    spectral_energy_ranges.append(energy_bin_edges)

        return datasets_3d_final, models_final, spectral_energy_ranges #return dataset, models and sed energy edges


class Dataset3DGeneration:
    """
    Separate class on 3D dataset creation based on the config or
    AsgardpyConfig information provided on the 3D dataset and the target source.

    Runs the following steps:
    1. Read the DL3 files of 3D datasets into gammapy readable objects.
    2. Create the base counts Map.
    3. Prepare standard data reduction using the parameters passed in the config
    for 3D datasets.
    4. Generate the final dataset.
    """

    def __init__(self, log, config_3d_dataset, config_full, key_name):
        self.instrument_name = config_3d_dataset.name
        self.config_3d_dataset_io = config_3d_dataset.io
        self.config_3d_dataset_info = config_3d_dataset.dataset_info
        self.key_name = key_name
        self.log = log

        # For updating the main config file with target source position
        # information if necessary.
        self.config_full = config_full
        self.config_target = config_full.target
        self.model = self.config_target.components.spectral

        self.exclusion_regions = []
        self.target_full_model = None

    def run(self):
        # First check for the given file list if they are readable or not.
        file_list = self.read_to_objects(self.model, self.key_name)

        # Start preparing objects to create the counts map
        self.set_energy_dispersion_matrix()
        self.load_events(file_list["events_file"])
        self.get_source_skycoord()

        # Create the Counts Map
        self._counts_map()

        # Create any dataset reduction makers or mask
        #self.add_source_to_exclusion_region()
        self._create_exclusion_mask()
        self._set_edisp_interpolator()
        self._set_exposure_interpolator()
        self._generate_diffuse_background_cutout()

        # Generate the final dataset
        dataset = self.generate_dataset() #self.key_name
        #self.log.info(f"Npred signal for the dataset {dataset.npred_signal().data.sum()}")
        #self.log.info(f"Mask image {dataset.mask_image}")
        #self.log.info(f"_Geom of the dataset {dataset._geom}")

        return dataset, self.list_sources, self.exclusion_mask # return MapDataset and list of models - target + bkg

    def read_to_objects(self, model, key_name):
        """
        For each key type of files, read the files to get the required
        Gammapy objects for further analyses.
        """
        lp_is_intrinsic = model.model_name == "LogParabola"
        #self.log.info(f"Is the model with intrinisic LP?: {model.model_name}, {lp_is_intrinsic}")
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
                self.get_source_pos_from_3d_dataset()

        return file_list

    def get_source_pos_from_3d_dataset(self):
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

    def get_base_objects(self, dl3_dir_dict, model, key, dl3_type, file_list):
        """
        For a DL3 files type and tag of the 'mode of observations' (FRONT or
        BACK for Fermi-LAT), read the files to appropriate Object type for
        further analysis.
        """
        dl3_info = DL3Files(dl3_dir_dict, model, file_list, log=self.log)
        file_list = dl3_info.prepare_lat_files(key, file_list)

        if dl3_type.lower() == "lat":
            self.exposure = Map.read(file_list["expmap_file"])
            self.psf = PSFMap.read(file_list["psf_file"], format="gtpsf")
            self.drmap = fits.open(file_list["edrm_file"])
            self.edisp_kernel = self.set_energy_dispersion_matrix()
            return file_list, [self.exposure, self.psf, self.drmap, self.edisp_kernel]

        elif dl3_type.lower() == "lat-aux":
            self.diff_gal = Map.read(file_list["diff_gal_file"])
            self.diff_iso = create_iso_diffuse_skymodel(file_list["iso_file"], key)

            return file_list, [self.diff_gal, self.diff_iso]
        else:
            return file_list, []

    def set_energy_axes(self):
        """
        Get the energy axes from the given Detector Response Matrix file.
        """
        energy_lo = self.drmap["DRM"].data["ENERG_LO"] * u.MeV
        energy_hi = self.drmap["DRM"].data["ENERG_HI"] * u.MeV

        energy_axis = MapAxis.from_energy_edges(np.append(energy_lo[0], energy_hi))
        energy_axis_true = energy_axis.copy(name="energy_true")

        return energy_axis, energy_axis_true

    def set_energy_dispersion_matrix(self):
        """
        Generate the Energy Dispersion Kernel from the given Detector Response
        Matrix file.
        """
        self.energy_axis, self.energy_axis_true = self.set_energy_axes()
        drm = self.drmap["DRM"].data["MATRIX"]
        drm_matrix = np.array(list(drm))

        self.edisp_kernel = EDispKernel(
            axes=[self.energy_axis_true, self.energy_axis], data=drm_matrix
        )

    def load_events(self, events_file):
        """
        Loading the events files for the specific "Key" into an EventList
        object and the GTI information into a GTI object.

        Based on any time intervals selection, filter out the events and gti
        accordingly.
        """
        self.event_fits = fits.open(events_file)
        self.events = EventList.read(events_file)
        self.gti = GTI.read(events_file)

        obs_time = self.config_3d_dataset_info.obs_time
        if obs_time.intervals[0].start is not None:
            t_start = Time(obs_time.intervals[0].start, format=obs_time.format)
            t_stop = Time(obs_time.intervals[0].stop, format=obs_time.format)
            time_intervals = [t_start, t_stop]

            self.events = self.events.select_time(time_intervals)
            self.gti = self.gti.select_time(time_intervals)

    def get_source_skycoord(self):
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

        self.source_pos = SkyCoord(ra_pos, dec_pos, unit="deg", frame="fk5")

    def get_list_objects(self, aux_path, xml_file, lp_is_intrinsic=False):
        """
        Read from the XML file to enlist the various objects and get their
        SkyModels
        """
        self.list_sources = []

        with open(xml_file) as f:
            data = xmltodict.parse(f.read())["source_library"]["source"]
            self.list_of_sources_final = [source["@name"] for source in data]

        for source in data:
            source_name = source["@name"]
            if source_name == "IsoDiffModel":
                source = self.diff_iso
            elif source_name == "GalDiffModel":
                source = create_gal_diffuse_skymodel(self.diff_gal)
            else:
                source, is_target_source = create_source_skymodel(
                    self.config_target, source, aux_path, lp_is_intrinsic
                )
            if is_target_source:
                self.target_full_model = source
                self.list_sources.insert(0, source)
            else:
                self.list_sources.append(source)

    def _counts_map(self):
        """
        Generate the counts Map object and fill it with the events' RA-Dec
        position, Energy and Time information.
        """
        self.counts_map = Map.create(
            skydir=self.source_pos,
            npix=(self.exposure.geom.npix[0][0], self.exposure.geom.npix[1][0]),
            proj="TAN",
            frame="fk5",
            binsz=self.exposure.geom.pixel_scales[0],
            axes=[self.energy_axis],
            dtype=float,
        )
        self.counts_map.fill_by_coord(
            {"skycoord": self.events.radec, "energy": self.events.energy, "time": self.events.time}
        )

    def _generate_diffuse_background_cutout(self):
        """
        Perform a cutout of the Diffuse background model with respect to the
        counts map geom (may improve fitting speed?).

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
        #self.list_sources.append(self.diff_gal_cutout)
        for k, m in enumerate(self.list_sources):
            if m.name == "diffuse-iem":
                self.list_sources[k] = self.diff_gal_cutout
        #print(Models(self.list_sources)["diffuse-iem"])

    def _set_edisp_interpolator(self):
        """
        Get the Energy Dispersion Kernel interpolated along true and
        reconstructed energy of the real counts.
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
        Generate an Exclusion Mask for the final MapDataset and also select
        the region for the list of Models using the excluded regions.
        """
        excluded_geom = self.counts_map.geom.copy()
        exclusion_params = self.config_3d_dataset_info.background.exclusion
        excluded_regions_list = []
        selected_regions_list = []

        if len(exclusion_params["regions"]) != 0:
            self.log.info("Using the background region from config for exclusion mask")
            for region in exclusion_params["regions"]:
                if region["name"] == "None":
                    coord = region["position"]
                    #print(coord)
                    center_ex = SkyCoord(
                        u.Quantity(coord["lon"]), u.Quantity(coord["lat"]), frame=coord["frame"]
                    ).icrs
                else:
                    #print(region["name"])
                    center_ex = SkyCoord.from_name(region["name"])
                #print(center_ex, region["parameters"]["rad_0"], region["parameters"]["rad_1"])

                # Generalize?
                excluded_region = CircleAnnulusSkyRegion(
                    center=center_ex,
                    inner_radius=u.Quantity(region["parameters"]["rad_0"]),
                    outer_radius=u.Quantity(region["parameters"]["rad_1"]),
                )
                selected_region = CircleSkyRegion(
                    center=center_ex,
                    radius=u.Quantity(region["parameters"]["rad_0"])
                )
                excluded_regions_list.append(excluded_region)
                selected_regions_list.append(selected_region)
            #print(excluded_regions_list, selected_regions_list)
            self.exclusion_mask = ~excluded_geom.region_mask(excluded_regions_list)

        elif len(self.exclusion_regions) == 0:
            self.log.info("Creating empty/dummy exclusion region")
            pos = SkyCoord(0, 90, unit="deg")
            exclusion_region = CircleSkyRegion(pos, 0.00001 * u.deg)
            excluded_regions_list.append(exclusion_region)
            self.exclusion_mask = ~excluded_geom.region_mask(excluded_regions_list)
        else:
            self.log.info("Creating exclusion region")
            self.exclusion_mask = ~excluded_geom.region_mask(self.exclusion_regions)
        #print("Excluded regions:", self.exclusion_regions)
        #print(selected_regions_list)
        #print("Radius of circular region around the selected center", selected_regions_list[0].radius)
        self.list_sources = Models(self.list_sources)
        new_models = []
        for m in self.list_sources:
            if "diffuse" not in m.name:
                sep = m.position.separation(center_ex)
                print(m.name, sep)
                if sep < selected_regions_list[0].radius:
                    new_models.append(m)
                else:
                    m.freeze()
                    new_models.append(m)
            else:
                new_models.append(m)
        new_models = Models(new_models)
        #print(new_models, new_models.names)
        #print("Models within the selected regions: ", self.list_sources.select_region(selected_regions_list))
        print("Separation of models location from selected center", new_models.positions.separation(center_ex))
        self.list_sources = new_models
        #self.list_sources = Models(self.list_sources).select_region(selected_regions_list[0])

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

    def generate_dataset(self): #, key_name
        """
        Generate MapDataset for the given Instrument files using the Counts Map,
        IRFs and Models objects.
        """

        try:
            mask_safe = self.exclusion_mask
            print("Using the exclusion mask to create a safe mask")
        except Exception:
            print("Other method of creating safe mask")
            mask_bool = np.zeros(self.counts_map.geom.data_shape).astype("bool")
            mask_safe = Map.from_geom(self.counts_map.geom, mask_bool)
            mask_safe.data = np.asarray(mask_safe.data == 0, dtype=bool)

        edisp = EDispKernelMap.from_edisp_kernel(self.edisp_interp_kernel)
        #print(mask_safe)
        #self.list_sources = Models(self.list_sources)
        #for model in self.list_sources:
        #    print(model.names)
        #    model.datasets_names = [f"Fermi-LAT_{key_name}"]
        #self.log.info(f"List of models being added {self.list_sources}")
        dataset = MapDataset(
            #models=self.list_sources,
            counts=self.counts_map,
            gti=self.gti,
            exposure=self.exposure_interp,
            psf=self.psf,
            edisp=edisp,
            mask_safe=mask_safe,
            name=f"{self.instrument_name}_{self.key_name}",
        )
        return dataset
