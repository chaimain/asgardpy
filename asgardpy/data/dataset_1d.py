"""
Generating 1D Datasets from given Instrument DL3 data
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
    ReflectedRegionsBackgroundMaker,
    ReflectedRegionsFinder,
    SafeMaskMaker,
    SpectrumDatasetMaker,
    WobbleRegionsFinder,
)
from gammapy.maps import MapAxis, RegionGeom, WcsGeom
from gammapy.modeling.models import Models
from regions import CircleSkyRegion, PointSkyRegion

from asgardpy.data.base import AnalysisStepBase, BaseConfig
from asgardpy.data.geom import EnergyAxisConfig, GeomConfig, SpatialPointConfig
from asgardpy.data.reduction import (
    BackgroundConfig,
    MapSelectionEnum,
    ObservationsConfig,
    ReductionTypeEnum,
    SafeMaskConfig,
)
from asgardpy.data.target import set_models
from asgardpy.io.io import DL3Files, InputConfig

__all__ = [
    "Dataset1DConfig",
    "Dataset1DBaseConfig",
    "Dataset1DConfig",
    "Dataset1DGeneration",
    "Datasets1DAnalysisStep",
]

log = logging.getLogger(__name__)


class Dataset1DInfoConfig(BaseConfig):
    name: str = "dataset-name"
    geom: GeomConfig = GeomConfig()
    observation: ObservationsConfig = ObservationsConfig()
    background: BackgroundConfig = BackgroundConfig()
    safe_mask: SafeMaskConfig = SafeMaskConfig()
    on_region: SpatialPointConfig = SpatialPointConfig()
    containment_correction: bool = True
    map_selection: List[MapSelectionEnum] = []
    spectral_energy_range: EnergyAxisConfig = EnergyAxisConfig()


class Dataset1DBaseConfig(BaseConfig):
    # stack: bool = True
    name: str = "Instrument-name"
    io: List[InputConfig] = [InputConfig()]
    dataset_info: Dataset1DInfoConfig = Dataset1DInfoConfig()


class Dataset1DConfig(BaseConfig):
    type: ReductionTypeEnum = ReductionTypeEnum.spectrum
    instruments: List[Dataset1DBaseConfig] = [Dataset1DBaseConfig()]


class Datasets1DAnalysisStep(AnalysisStepBase):
    """
    From the given config information, prepare the full list of 1D datasets.
    """

    tag = "datasets-1d"

    def _run(self):
        # Iterate over all instrument information given:
        instruments_list = self.config.dataset1d.instruments
        self.log.info(f"{len(instruments_list)} number of 1D Datasets given")

        datasets_1d_final = Datasets()
        spectral_energy_ranges = []

        for i in np.arange(len(instruments_list)):
            self.config_1d_dataset = instruments_list[i]

            generate_1d_dataset = Dataset1DGeneration(
                self.log, self.config_1d_dataset, self.config.target
            )

            dataset = generate_1d_dataset.run()

            # Get the spectral energy information for each Instrument Dataset
            energy_range = self.config_1d_dataset.dataset_info.spectral_energy_range
            energy_bin_edges = MapAxis.from_energy_bounds(
                energy_min=u.Quantity(energy_range.min),
                energy_max=u.Quantity(energy_range.max),
                nbin=int(energy_range.nbins),
                per_decade=True,
            ).edges

            if self.config.general.stacked_dataset:
                dataset = dataset.stack_reduce(name=self.config_1d_dataset.name)
                print(dataset.name)
                print(dataset.models, type(dataset.models))
                dataset.models = Models(models=None)
                #dataset = set_models(config=self.config.target, datasets=dataset)
                #print(dataset.models.names)
                #print(dataset.models.datasets_names)
                datasets_1d_final.append(dataset)
                spectral_energy_ranges.append(energy_bin_edges)
            else:
                for data in dataset:
                    #print(data.models.names, f"{self.config_1d_dataset.name} {data.models.names[0]}")
                    datasets_1d_final.append(data)
                    spectral_energy_ranges.append(energy_bin_edges)

        return datasets_1d_final, spectral_energy_ranges


class Dataset1DGeneration:
    """
    Separate class on 1D dataset creation based on the config or
    AsgardpyConfig information provided on the 1D dataset and the target source.

    Runs the following steps:
    1. Read the DL3 files of 1D datasets into gammapy readable objects.
    2. Perform any Observation selection.
    3. Create the base dataset template, including the main counts geometry.
    4. Prepare standard data reduction using the parameters passed in the config
    for 1D datasets.
    5. Generate the final dataset.
    """

    def __init__(self, log, config_1d_dataset, config_target):
        self.config_1d_dataset_io = config_1d_dataset.io
        self.log = log
        self.config_1d_dataset_info = config_1d_dataset.dataset_info
        self.config_target = config_target

        # Only 1 DL3 type of file.
        self.datasets = Datasets()
        self.dl3_dir_dict = self.config_1d_dataset_io[0]
        self.model = config_target.components.spectral

    def run(self):
        # First check for the given file list if they are readable or not.
        file_list = {}
        dl3_info = DL3Files(self.dl3_dir_dict, self.model, file_list, log=self.log)
        dl3_info.list_dl3_files()

        # Create a Datastore object to select Observations object
        self.datastore = DataStore.from_dir(self.dl3_dir_dict.input_dir)

        # Applying all provided filters to get the Observations object
        self.get_filtered_observations()

        # Create the main counts geometry
        self.dataset_template = self.generate_geom()

        # Get all the Dataset reduction makers
        self.dataset_maker, self.bkg_maker, self.safe_maker = self.get_reduction_makers()

        # Produce the final Dataset
        self.generate_dataset()

        return self.datasets

    def get_filtered_observations(self):
        """
        From the Datastores object, apply any observation filters provided in
        the config file create the Observations object.
        """
        # Could be generalized along with the same in dataset_3d
        obs_time = self.config_1d_dataset_info.observation.obs_time
        obs_list = self.config_1d_dataset_info.observation.obs_ids

        obs_table = self.datastore.obs_table.group_by("OBS_ID")
        observation_mask = np.ones(len(obs_table), dtype=bool)

        # Filter using the Time interval range provided
        if obs_time.intervals[0].start is not None:
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
        self.observations = self.datastore.get_observations(
            filtered_obs_ids, required_irf=irfs_selected
        )

    def generate_geom(self):
        """
        Generate from a given or target source position, provided in astropy's
        SkyCoord readable values, the geometry of the ON events and the axes
        information on reco energy and true energy, a dataset can be defined.
        """
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
        if ~hasattr(given_on_geom, "radius"):
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
        reco_energy_from_config = self.config_1d_dataset_info.geom.axes.energy
        energy_axis = MapAxis.from_energy_bounds(
            energy_min=u.Quantity(reco_energy_from_config.min),
            energy_max=u.Quantity(reco_energy_from_config.max),
            nbin=int(reco_energy_from_config.nbins),
            per_decade=True,
            name="energy",
        )
        true_energy_from_config = self.config_1d_dataset_info.geom.axes.energy_true
        true_energy_axis = MapAxis.from_energy_bounds(
            energy_min=u.Quantity(true_energy_from_config.min),
            energy_max=u.Quantity(true_energy_from_config.max),
            nbin=int(true_energy_from_config.nbins),
            per_decade=True,
            name="energy_true",
        )

        # Main geom and template Spectrum Dataset
        geom = RegionGeom.create(region=on_region, axes=[energy_axis])
        dataset_template = SpectrumDataset.create(geom=geom, energy_axis_true=true_energy_axis)

        return dataset_template

    def get_reduction_makers(self):
        """
        Get Makers for Dataset creation, Background and Safe Energy Mask
        reduction.
        Maybe make them into 3 distinct funcrions.
        """
        # Spectrum Dataset Maker
        dataset_maker = SpectrumDatasetMaker(
            containment_correction=self.config_1d_dataset_info.containment_correction,
            selection=self.config_1d_dataset_info.map_selection,
        )

        # Background reduction maker
        bkg_config = self.config_1d_dataset_info.background

        # Exclusion mask
        if bkg_config.exclusion:
            if bkg_config.exclusion["name"] == "None":
                coord = bkg_config.exclusion["position"]
                center_ex = SkyCoord(
                    u.Quantity(coord["lon"]), u.Quantity(coord["lat"]), frame=coord["frame"]
                ).icrs
            else:
                center_ex = SkyCoord.from_name(bkg_config.exclusion["name"])

            excluded_region = CircleSkyRegion(
                center=center_ex, radius=u.Quantity(bkg_config.exclusion["region_radius"])
            )
        else:
            excluded_region = None

        # Needs to be united with other Geometry creation functions, into a separate class
        # Also make these geom parameters also part of the config requirements
        excluded_geom = WcsGeom.create(
            npix=(125, 125), binsz=0.05, skydir=center_ex, proj="TAN", frame="icrs"
        )
        exclusion_mask = ~excluded_geom.region_mask([excluded_region])

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

        # Safe Energy Mask Maker
        safe_config = self.config_1d_dataset_info.safe_mask
        pars = safe_config.parameters

        if len(safe_config.methods) != 0:
            if "custom-mask" not in safe_config.methods:
                pos = SkyCoord(
                    u.Quantity(pars.position["lon"]),
                    u.Quantity(pars.position["lat"]),
                    frame=pars.position["frame"],
                )
                safe_maker = SafeMaskMaker(
                    methods=safe_config.methods,
                    aeff_percent=pars.aeff_percent,
                    bias_percent=pars.bias_percent,
                    position=pos,
                    fixed_offset=pars.fixed_offset,
                    offset_max=pars.offset_max,
                )
            else:
                safe_maker = None
        else:
            safe_maker = None

        return dataset_maker, bkg_maker, safe_maker

    def generate_dataset(self):
        """
        From the given Observations and various Makers, produce the
        DatasetOnOff object and append it to the final Datasets object.
        """
        for obs in self.observations:
            dataset = self.dataset_maker.run(self.dataset_template.copy(name=str(obs.obs_id)), obs)

            dataset_on_off = self.bkg_maker.run(dataset, obs)
            # Necessary meta information addition?
            dataset_on_off.meta_table["SOURCE"] = self.config_target.source_name

            safe_cfg = self.config_1d_dataset_info.safe_mask
            if "custom-mask" in safe_cfg.methods:
                pars = safe_cfg.parameters
                dataset_on_off.mask_safe = dataset_on_off.counts.geom.energy_mask(
                    energy_min=u.Quantity(pars["min"]), energy_max=u.Quantity(pars["max"])
                )
            elif len(safe_cfg.methods) != 0:
                dataset_on_off = self.safe_maker.run(dataset_on_off, obs)
            else:
                self.log.info(f"No safe mask applied for {obs.obs_id}")

            self.datasets.append(dataset_on_off)
