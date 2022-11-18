"""
Generating 1D Datasets from given Instrument DL3 data
"""

import logging

from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import PointSkyRegion, CircleSkyRegion

# from gammapy.analysis import Analysis, AnalysisConfig - no support for DL3 with RAD_MAX
from gammapy.data import DataStore
from gammapy.datasets import (
    Datasets,
    SpectrumDataset,
)
from gammapy.makers import (
    SafeMaskMaker,
    SpectrumDatasetMaker,
    ReflectedRegionsBackgroundMaker,
)
from gammapy.maps import MapAxis, RegionGeom, WcsGeom

from .io import DL3Files
from .dataset_3d import get_source_pos_from_3d_dataset

__all__ = ["Dataset1D", "Dataset1DIO", "Dataset1DInfo"]

log = logging.getLogger(__name__)


class Dataset1DIO(DL3Files):
    """
    Read the DL3 files of 1D datasets.
    """
    def __init__(self, config, instrument_idx=0):
        self.config_1d_dataset = config["Dataset1D"]["Instruments"][instrument_idx]
        self.config_1d_dataset_io = self.config_1d_dataset["IO"]
        self.datastore = None
        self.observations = None

        self.dl3_path = self.config_1d_dataset_io["input_dir"]
        model = self.config["Target_model"]["spectral"]["type"]
        dl3_type = self.config_1d_dataset["name"]

        super().__init__(self.dl3_path, model, dl3_type)

    def load_data(self):
        self.datastore = DataStore.from_dir(self.dl3_path)
        self.observations = self.datastore.get_observations(
            required_irf=self.config_1d_dataset_info["required_irfs"]
        )


class Dataset1DInfo:
    """
    Prepare standard data reduction using the parameters passed in the config
    for 1D datasets.
    """
    def __init__(self, config, instrument_idx=0):
        self.config_1d_dataset = config["Dataset1D"]["Instruments"][instrument_idx]
        self.config_1d_dataset_info = self.config_1d_dataset["DatasetInfo"]
        # self.config.set_logging()

    def generate_geom(self):
        """
        From a given or corrected target source position, provided in
        astropy's SkyCoord object, the geometry of the ON events and the
        dataset is defined.
        """
        target_config = self.config["Target_source"]

        if target_config["use_uniform_position"]:
            # Using the same target source position as that used for
            # the 3D datasets analysis
            src_pos = get_source_pos_from_3d_dataset()
        else:
            src_name = target_config["source_name"]
            if src_name is not None:
                src_pos = SkyCoord.from_name(src_name)
            else:
                src_pos = SkyCoord(
                    ra=u.Quantity(target_config["sky_position"]["ra"]),
                    dec=u.Quantity(target_config["sky_position"]["dec"])
                )

        given_on_geom = self.config_1d_dataset_info["base_geom"]["on_region"]
        if given_on_geom == PointSkyRegion.__name__:
            on_region = PointSkyRegion(src_pos)
            # Hack to allow for the joint fit
            # (otherwise pointskyregion.contains returns None)
            on_region.meta = {"include": False}

        elif given_on_geom == CircleSkyRegion.__name__:
            on_region = CircleSkyRegion(
                center=src_pos,
                radius=u.Quantity(
                    self.config_1d_dataset_info["base_geom"]["on_region_radius"]
                )
            )

        reco_energy_from_config = self.config_1d_dataset_info["energy_axes"]["reco_energy_axis"]
        energy_axis = MapAxis.from_energy_bounds(
            energy_min=u.Quantity(
                reco_energy_from_config["min"]
            ),
            energy_max=u.Quantity(
                reco_energy_from_config["max"]
            ),
            nbin=int(reco_energy_from_config["nbins"]),
            per_decade=True,
            name="energy"
        )
        true_energy_from_config = self.config_1d_dataset_info["energy_axes"]["true_energy_axis"]
        true_energy_axis = MapAxis.from_energy_bounds(
            energy_min=u.Quantity(
                true_energy_from_config["min"]
            ),
            energy_max=u.Quantity(
                true_energy_from_config["max"]
            ),
            nbin=int(true_energy_from_config["nbins"]),
            per_decade=True,
            name="energy_true"
        )
        geom = RegionGeom.create(
            region=on_region,
            axes=[energy_axis]
        )

        dataset_template = SpectrumDataset.create(
            geom=geom,
            energy_axis_true=true_energy_axis
        )
        return dataset_template

    def get_reduction_makers(self):
        """
        Get Makers for Dataset creation, Background and Safe Energy Mask
        reduction
        """
        # Spectrum Dataset Maker
        dataset_maker = SpectrumDatasetMaker(
            containment_correction=self.config_1d_dataset_info["containment_correction"],
            selection=self.config_1d_dataset_info["map_selection"]
        )

        # Background reduction maker
        bkg_config = self.config_1d_dataset_info["background"]

        # Exclusion mask
        if bkg_config["excluded"]["name"] is None:
            coord = bkg_config["excluded"]["region_coord"]
            center_ex = SkyCoord(
                u.Quantity(coord["gal_lon"]),
                u.Quantity(coord["gal_lat"]),
                frame="galactic"
            ).icrs
        else:
            center_ex = SkyCoord.from_name(bkg_config["excluded"]["name"])

        excluded_region = CircleSkyRegion(
            center=center_ex,
            radius=u.Quantity(bkg_config["excluded"]["region_radius"])
        )
        excluded_geom = WcsGeom.create(
            npix=(125, 125), binsz=0.05,
            skydir=center_ex, proj="TAN",
            frame="icrs"
        )
        exclusion_mask = ~excluded_geom.region_mask([excluded_region])

        if bkg_config["method"] == "reflected":
            bkg_maker = ReflectedRegionsBackgroundMaker(
                n_off_regions=int(bkg_config["wobble_off_regions"]),
                exclusion_mask=exclusion_mask
            )
        else:
            bkg_maker = None

        # Safe Energy Mask
        safe_config = self.config_1d_dataset_info["safe_energy_mask"]
        if not safe_config["method"]["custom"]:
            std_pars = safe_config["parameters"]["standard"]
            pos = SkyCoord(
                ra=u.Quantity(std_pars["position"]["ra"]),
                dec=u.Quantity(std_pars["position"]["dec"])
            )
            safe_maker = SafeMaskMaker(
                methods=safe_config["method"]["standard"],
                aeff_percent=std_pars["aeff_percent"],
                bias_percent=std_pars["bias_percent"],
                position=pos,
                fixed_offset=std_pars["fixed_offset"],
                offset_max=std_pars["offset_max"],
            )
        else:
            safe_maker = None

        return dataset_maker, bkg_maker, safe_maker


class Dataset1D(Dataset1DIO, Dataset1DInfo):
    """
    Using the Datastore and Observations generated after reading the DL3 files,
    and the various data reduction makers, generate 1D Datasets.
    """
    def __init__(self, config):
        io = Dataset1DIO(self, config, instrument_idx=0)
        info = Dataset1DInfo(self, config, instrument_idx=0)

        self.observations = io.load_data().observations
        self.datastore = io.load_data().datastore

        self.dataset_template = info.generate_geom()
        self.dataset_maker, self.bkg_maker, self.safe_maker = info().get_reduction_makers()

        self.datasets = Datasets()

    def generate_dataset(self):
        """
        From the given Observations and various Makers, produce the Datasets
        object.
        """
        for obs in self.observations:
            dataset = self.dataset_maker.run(
                self.dataset_template.copy(
                    name=str(obs.obs_id)
                ),
                obs
            )
            dataset_on_off = self.bkg_maker.run(dataset, obs)
            dataset_on_off.meta_table["SOURCE"] = self.config["Target_source"]["source_name"]

            safe_cfg = self.config_1d_dataset_info["safe_energy_mask"]
            if safe_cfg["method"]["custom"]:
                safe_en = safe_cfg["parameters"]["custom_energy_range"]
                dataset_on_off.mask_safe = dataset_on_off.counts.geom.energy_mask(
                    energy_min=u.Quantity(safe_en["min"]),
                    energy_max=u.Quantity(safe_en["max"])
                )
            else:
                dataset_on_off = self.safe_maker.run(dataset_on_off, obs)
            self.datasets.append(dataset_on_off)

        return self.datasets
