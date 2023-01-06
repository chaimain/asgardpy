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
from regions import CircleSkyRegion, PointSkyRegion

from asgardpy.data.base import AnalysisStepBase, BaseConfig
from asgardpy.data.dataset_3d import Dataset3DGeneration
from asgardpy.data.geom import GeomConfig, SpatialPointConfig
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
    Using the Datastore and Observations generated after reading the DL3 files,
    and the various data reduction makers, generate 1D Datasets.
    """

    tag = "datasets-1d"

    def _run(self):
        # Iterate over all instrument information given:
        instruments_list = self.config.dataset1d.instruments
        self.dataset_3d = self.config.dataset3d.instruments[0]
        self.log.info(f"{len(instruments_list)} number of 1D Datasets given")

        datasets_1d_final = Datasets()

        for i in np.arange(len(instruments_list)):
            self.config_1d_dataset = instruments_list[i]

            generate_1d_dataset = Dataset1DGeneration(
                self.config_1d_dataset, self.config.target, self.dataset_3d
            )
            dataset = generate_1d_dataset.run()
            # for data in dataset:
            dataset = dataset.stack_reduce(name=self.config_1d_dataset.name)
            dataset = set_models(config=self.config.target, datasets=dataset)
            datasets_1d_final.append(dataset)

        return datasets_1d_final


class Dataset1DGeneration:
    """
    Separate class on 1D dataset creation based on the config or
    AsgardpyConfig information provided on the 1D dataset and the target source.

    Runs the following steps:
    1. Read the DL3 files of 1D datasets into gammapy readable objects.
    2. Prepare standard data reduction using the parameters passed in the config
    for 1D datasets.
    3. Generate the final dataset.
    """

    def __init__(self, config_1d_dataset, config_target, config_3d_dataset):
        self.config_1d_dataset_io = config_1d_dataset.io
        self.dataset_3d_src_pos = config_3d_dataset
        self.config_1d_dataset_info = config_1d_dataset.dataset_info
        self.config_target = config_target
        # Only 1 DL3 type of file.
        self.datasets = Datasets()
        self.dl3_dir_dict = self.config_1d_dataset_io[0]
        self.model = config_target.components.spectral

    def run(self):
        file_list = {}
        dl3_info = DL3Files(self.dl3_dir_dict, self.model, file_list)
        dl3_info.list_dl3_files()

        self.datastore = DataStore.from_dir(self.dl3_dir_dict.input_dir)

        irfs_selected = self.config_1d_dataset_info.observation.required_irfs
        self.observations = self.datastore.get_observations(required_irf=irfs_selected)

        obs_time = self.config_1d_dataset_info.observation.obs_time
        # Could be generalized along with the same in dataset_3d
        if obs_time.intervals[0].start is not None:
            t_start = Time(obs_time.intervals[0].start, format=obs_time.format)
            t_stop = Time(obs_time.intervals[0].stop, format=obs_time.format)
            time_intervals = [t_start, t_stop]

            self.observations = self.observations.select_time([time_intervals])

        self.dataset_template = self.generate_geom()
        self.dataset_maker, self.bkg_maker, self.safe_maker = self.get_reduction_makers()
        datasets = self.generate_dataset()

        return datasets

    def generate_geom(self):
        """
        From a given or corrected target source position, provided in
        astropy's SkyCoord object, the geometry of the ON events and the
        axes information on reco energy and true energy, a dataset can be defined.
        """
        if self.config_target.use_uniform_position:
            # Using the same target source position as that used for
            # the 3D datasets analysis. Which one?
            dataset_3d = Dataset3DGeneration(
                self.dataset_3d_src_pos,  # Need to fix this
                self.config_target,
                self.dataset_3d_src_pos.dataset_info.key[0],
            )
            dataset_3d.read_to_objects(self.model, self.dataset_3d_src_pos.dataset_info.key[0])
            src_pos = dataset_3d.get_source_pos_from_3d_dataset()
        else:
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
        reduction
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

        return dataset_maker, bkg_maker, safe_maker

    def generate_dataset(self):
        """
        From the given Observations and various Makers, produce the Datasets
        object.
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
            else:
                dataset_on_off = self.safe_maker.run(dataset_on_off, obs)
            self.datasets.append(dataset_on_off)

        return self.datasets
