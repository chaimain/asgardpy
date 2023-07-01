"""
Main classes to define 1D Dataset Config, 1D Dataset Analysis Step and
to generate 1D Datasets from given Instruments' DL3 data from the config.
"""

import logging
from typing import List

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

# from gammapy.analysis import Analysis, AnalysisConfig - no support for DL3 with RAD_MAX
from gammapy.data import DataStore
from gammapy.datasets import Datasets, SpectrumDataset
from gammapy.makers import (
    DatasetsMaker,
    ReflectedRegionsBackgroundMaker,
    ReflectedRegionsFinder,
    SafeMaskMaker,
    SpectrumDatasetMaker,
    WobbleRegionsFinder,
)
from gammapy.maps import RegionGeom, WcsGeom
from regions import CircleAnnulusSkyRegion, CircleSkyRegion, PointSkyRegion

from asgardpy.base import (
    AnalysisStepBase,
    BackgroundConfig,
    BaseConfig,
    GeomConfig,
    MapAxesConfig,
    MapSelectionEnum,
    ObservationsConfig,
    ReductionTypeEnum,
    SafeMaskConfig,
    SkyPositionConfig,
    get_energy_axis,
)
from asgardpy.io import DL3Files, InputConfig

__all__ = [
    "Datasets1DAnalysisStep",
    "Dataset1DBaseConfig",
    "Dataset1DConfig",
    "Dataset1DGeneration",
    "Dataset1DInfoConfig",
]

log = logging.getLogger(__name__)


# Defining various components of 1D Dataset Config section
class Dataset1DInfoConfig(BaseConfig):
    name: str = "dataset-name"
    geom: GeomConfig = GeomConfig()
    observation: ObservationsConfig = ObservationsConfig()
    background: BackgroundConfig = BackgroundConfig()
    safe_mask: SafeMaskConfig = SafeMaskConfig()
    on_region: SkyPositionConfig = SkyPositionConfig()
    containment_correction: bool = True
    map_selection: List[MapSelectionEnum] = []
    spectral_energy_range: MapAxesConfig = MapAxesConfig()


class Dataset1DBaseConfig(BaseConfig):
    name: str = "Instrument-name"
    io: List[InputConfig] = [InputConfig()]
    dataset_info: Dataset1DInfoConfig = Dataset1DInfoConfig()


class Dataset1DConfig(BaseConfig):
    type: ReductionTypeEnum = ReductionTypeEnum.spectrum
    instruments: List[Dataset1DBaseConfig] = [Dataset1DBaseConfig()]


# The main Analysis Step
class Datasets1DAnalysisStep(AnalysisStepBase):
    """
    From the given config information, prepare the full list of 1D datasets,
    iterating over all the Instruments' information by running the
    Dataset1DGeneration function.
    """

    tag = "datasets-1d"

    def _run(self):
        instruments_list = self.config.dataset1d.instruments
        self.log.info(f"{len(instruments_list)} number of 1D Datasets given")

        datasets_1d_final = Datasets()
        instrument_spectral_info = {"name": [], "spectral_energy_ranges": []}

        # Iterate over all instrument information given:
        for i in np.arange(len(instruments_list)):
            config_1d_dataset = instruments_list[i]
            instrument_spectral_info["name"].append(config_1d_dataset.name)

            generate_1d_dataset = Dataset1DGeneration(self.log, config_1d_dataset, self.config)
            dataset = generate_1d_dataset.run()

            # Get the spectral energy information for each Instrument Dataset
            energy_axes = config_1d_dataset.dataset_info.spectral_energy_range
            if len(energy_axes.axis_custom.edges) > 0:
                energy_bin_edges = get_energy_axis(energy_axes, only_edges=True, custom_range=True)
            else:
                energy_bin_edges = get_energy_axis(
                    energy_axes,
                    only_edges=True,
                )
            instrument_spectral_info["spectral_energy_ranges"].append(energy_bin_edges)

            if self.config.general.stacked_dataset:
                dataset = dataset.stack_reduce(name=config_1d_dataset.name)
                datasets_1d_final.append(dataset)
            else:
                for data in dataset:
                    datasets_1d_final.append(data)

        return (
            datasets_1d_final,
            None,
            instrument_spectral_info,
        )


class Dataset1DGeneration:
    """
    Class for 1D dataset creation based on the config or AsgardpyConfig
    information provided on the 1D dataset and the target source.

    Runs the following steps:

    1. Read the DL3 files of 1D datasets into DataStore object.

    2. Perform any Observation selection, based on Observation IDs or time intervals.

    3. Create the base dataset template, including the main counts geometry.

    4. Prepare standard data reduction makers using the parameters passed in the config.

    5. Generate the final dataset.
    """

    def __init__(self, log, config_1d_dataset, config_full):
        self.config_1d_dataset_io = config_1d_dataset.io
        self.log = log
        self.config_1d_dataset_info = config_1d_dataset.dataset_info
        self.config_target = config_full.target
        self.n_jobs = config_full.general.n_jobs
        self.multi = config_full.general.parallel_backend
        self.exclusion_regions = []
        self.datasets = Datasets()

    def run(self):
        # First check for the given file list if they are readable or not.
        file_list = {}
        dl3_info = DL3Files(
            self.config_1d_dataset_io[0],
            file_list,
            log=self.log,
        )
        dl3_info.list_dl3_files()

        # Create a Datastore object to select Observations object
        datastore = DataStore.from_dir(self.config_1d_dataset_io[0].input_dir)

        # Applying all provided filters to get the Observations object
        observations = self.get_filtered_observations(datastore)

        # Create the main counts geometry
        dataset_template = self.generate_geom()

        # Get all the Dataset reduction makers
        dataset_maker = SpectrumDatasetMaker(
            containment_correction=self.config_1d_dataset_info.containment_correction,
            selection=self.config_1d_dataset_info.map_selection,
        )
        bkg_maker = self.get_bkg_maker()
        safe_maker = self.get_safe_mask_maker()

        # Produce the final Dataset
        self.generate_dataset(observations, dataset_template, dataset_maker, bkg_maker, safe_maker)

        return self.datasets

    def get_filtered_observations(self, datastore):
        """
        From the DataStore object, apply any observation filters provided in
        the config file to create and return an Observations object.
        """
        # Could be generalized along with the same in dataset_3d
        obs_time = self.config_1d_dataset_info.observation.obs_time
        obs_list = self.config_1d_dataset_info.observation.obs_ids

        obs_table = datastore.obs_table.group_by("OBS_ID")
        observation_mask = np.ones(len(obs_table), dtype=bool)

        # Filter using the Time interval range provided
        if obs_time.intervals[0].start != Time("0", format="mjd"):
            t_start = Time(obs_time.intervals[0].start, format=obs_time.format)
            t_stop = Time(obs_time.intervals[0].stop, format=obs_time.format)

            full_time = []
            for obs in obs_table:
                full_time.append(Time(f"{obs['DATE-OBS']} {obs['TIME-OBS']}", format="iso"))
            full_time = np.array(full_time)

            time_min_mask = full_time > t_start
            time_max_mask = full_time < t_stop

            observation_mask *= time_min_mask * time_max_mask

        filtered_obs_ids = obs_table[observation_mask]["OBS_ID"].data

        # Filter using the given list of observation ids provided
        if len(obs_list) != 0:
            filtered_obs_ids = np.intersect1d(
                filtered_obs_ids, np.array(obs_list), assume_unique=True
            )
        self.log.info(f"Observation ID list selected: {filtered_obs_ids}")

        # IRFs selection
        irfs_selected = self.config_1d_dataset_info.observation.required_irfs
        observations = datastore.get_observations(filtered_obs_ids, required_irf=irfs_selected)

        return observations

    def generate_geom(self):
        """
        Generate from a given target source position, the geometry of the ON
        events and the axes information on reco energy and true energy,
        for which the 1D dataset can be defined.
        """
        # Get source position as astropy SkyCoord object
        src_name = self.config_target.source_name
        if src_name is not None:
            src_pos = SkyCoord.from_name(src_name)
        else:
            src_pos = SkyCoord(
                u.Quantity(self.config_target.sky_position.lon),
                u.Quantity(self.config_target.sky_position.lat),
                frame=self.config_target.sky_position.frame,
            )

        # Defining the ON region's geometry
        given_on_geom = self.config_1d_dataset_info.on_region
        if given_on_geom.radius == 0 * u.deg:
            on_region = PointSkyRegion(src_pos)
            # Hack to allow for the joint fit
            # (otherwise pointskyregion.contains returns None)
            on_region.meta = {"include": False}

        else:
            on_region = CircleSkyRegion(
                center=src_pos,
                radius=u.Quantity(given_on_geom.radius),
            )

        # Defining the energy axes
        axes_list = self.config_1d_dataset_info.geom.axes
        for axes_ in axes_list:
            energy_axis = get_energy_axis(axes_)
            # Main geom and template Spectrum Dataset
            if axes_.name == "energy":
                geom = RegionGeom.create(region=on_region, axes=[energy_axis])

            if axes_.name == "energy_true":
                dataset_template = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis)

        return dataset_template

    def get_bkg_maker(self):
        """
        Generate Background reduction maker by including an Exclusion mask
        with any exclusion regions' information on a Map geometry using the
        information provided in the config.
        """
        bkg_config = self.config_1d_dataset_info.background
        exclusion_params = bkg_config.exclusion

        # Exclusion mask
        if len(exclusion_params.regions) != 0:
            for region in exclusion_params.regions:
                if region.name == "":
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
                self.exclusion_regions.append(excluded_region)
        else:
            center_ex = SkyCoord(
                u.Quantity(self.config_target.sky_position.lon),
                u.Quantity(self.config_target.sky_position.lat),
                frame=self.config_target.sky_position.frame,
            )
            self.exclusion_regions = []

        geom_config = self.config_1d_dataset_info.geom

        bin_size = geom_config.wcs.binsize.to_value(u.deg)
        width_ = geom_config.wcs.map_frame_shape.width.to_value(u.deg)
        width_in_pixel = int(width_ / bin_size)
        height_ = geom_config.wcs.map_frame_shape.height.to_value(u.deg)
        height_in_pixel = int(height_ / bin_size)

        excluded_geom = WcsGeom.create(
            npix=(width_in_pixel, height_in_pixel),
            binsz=bin_size,
            skydir=center_ex,
            proj=geom_config.wcs.proj,
            frame="icrs",
        )
        if len(self.exclusion_regions) > 0:
            exclusion_mask = ~excluded_geom.region_mask(self.exclusion_regions)
        else:
            exclusion_mask = None

        # Background reduction maker. Need to generalize further.
        if bkg_config.method == "reflected":
            if bkg_config.region_finder_method == "wobble":
                region_finder = WobbleRegionsFinder(**bkg_config.parameters)
            elif bkg_config.region_finder_method == "reflected":
                region_finder = ReflectedRegionsFinder(**bkg_config.parameters)

            bkg_maker = ReflectedRegionsBackgroundMaker(
                region_finder=region_finder, exclusion_mask=exclusion_mask
            )
        else:
            bkg_maker = None

        return bkg_maker

    def get_safe_mask_maker(self):
        """
        Generate Safe mask reduction maker as per the selections provided in
        the config.
        """
        safe_config = self.config_1d_dataset_info.safe_mask
        pars = safe_config.parameters

        if len(safe_config.methods) != 0:
            if "custom-mask" not in safe_config.methods:
                safe_maker = SafeMaskMaker(methods=safe_config.methods, **pars)
            else:
                safe_maker = None
        else:
            safe_maker = None

        return safe_maker

    def generate_dataset(
        self, observations, dataset_template, dataset_maker, bkg_maker, safe_maker
    ):
        """
        From the given Observations, Dataset Template and various Makers,
        use the multiprocessing method with DatasetsMaker and update the
        datasets accordingly.
        """
        if safe_maker:
            makers = [dataset_maker, safe_maker, bkg_maker]
        else:
            makers = [dataset_maker, bkg_maker]

        datasets_maker = DatasetsMaker(
            makers,
            stack_datasets=False,
            n_jobs=self.n_jobs,
            parallel_backend=self.parallel_backend,
        )
        self.datasets = datasets_maker.run(dataset_template, observations)

        safe_cfg = self.config_1d_dataset_info.safe_mask
        pars = safe_cfg.parameters

        for data, obs in zip(self.datasets, observations):
            # Rename the datasets using the appropriate Obs ID
            data._name = obs.obs_id

            # Use custom safe energy mask
            if safe_maker is None:
                data.mask_safe = data.counts.geom.energy_mask(
                    energy_min=u.Quantity(pars["min"]),
                    energy_max=u.Quantity(pars["max"]),
                )
