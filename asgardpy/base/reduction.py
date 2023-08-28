"""
Classes containing the Dataset Reduction config parameters for the high-level
interface and also various functions to support in Dataset Reduction and
creating of the appropriate DL4 dataset.
"""

from enum import Enum
from typing import List

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from gammapy.catalog import CATALOG_REGISTRY
from gammapy.data import DataStore
from gammapy.datasets import MapDataset, SpectrumDataset
from gammapy.makers import (
    DatasetsMaker,
    FoVBackgroundMaker,
    MapDatasetMaker,
    ReflectedRegionsBackgroundMaker,
    ReflectedRegionsFinder,
    RingBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
    WobbleRegionsFinder,
)
from regions import CircleAnnulusSkyRegion, CircleSkyRegion

from asgardpy.base.base import AngleType, BaseConfig, PathType, TimeIntervalsConfig
from asgardpy.base.geom import SkyPositionConfig, get_energy_axis

__all__ = [
    "BackgroundConfig",
    "BackgroundMethodEnum",
    "BackgroundRegionFinderMethodEnum",
    "ExclusionRegionsConfig",
    "generate_dl4_dataset",
    "get_bkg_maker",
    "get_dataset_maker",
    "get_dataset_reference",
    "get_exclusion_region_mask",
    "get_filtered_observations",
    "get_safe_mask_maker",
    "MapSelectionEnum",
    "ObservationsConfig",
    "ReductionTypeEnum",
    "ReflectedRegionFinderConfig",
    "RegionsConfig",
    "RequiredHDUEnum",
    "SafeMaskConfig",
    "SafeMaskMethodsEnum",
    "WobbleRegionsFinderConfig",
]


# Basic Components to define the various Dataset Reduction Maker Config
class ReductionTypeEnum(str, Enum):
    """
    Config section for list of DL3 Dataset Reduction methods used in Gammapy.
    """

    spectrum = "1d"
    cube = "3d"


class RequiredHDUEnum(str, Enum):
    """
    Config section for list of HDU objects required for reading the DL3 file
    into a Dataset object.
    """

    aeff = "aeff"
    bkg = "bkg"
    edisp = "edisp"
    psf = "psf"
    rad_max = "rad_max"
    point_like = "point-like"
    full_enclosure = "full-enclosure"


class ObservationsConfig(BaseConfig):
    """
    Config section for getting main information for creating an Observations
    object.
    """

    obs_ids: List[int] = []
    obs_file: PathType = PathType(".")
    obs_time: TimeIntervalsConfig = TimeIntervalsConfig()
    obs_cone: SkyPositionConfig = SkyPositionConfig()
    required_irfs: List[RequiredHDUEnum] = [RequiredHDUEnum.aeff]


class BackgroundMethodEnum(str, Enum):
    """
    Config section for list of methods for creating Background Reduction
    Makers object.
    """

    reflected = "reflected"
    fov = "fov_background"
    ring = "ring"


class BackgroundRegionFinderMethodEnum(str, Enum):
    """
    Config section for list of background region finder methods for creating
    the Background Reduction Makers object.
    """

    reflected = "reflected"
    wobble = "wobble"


class ReflectedRegionFinderConfig(BaseConfig):
    """
    Config section for getting main information for creating a
    ReflectedRegionsFinder object.
    """

    angle_increment: AngleType = 0.01 * u.deg
    min_distance: AngleType = 0.1 * u.deg
    min_distance_input: AngleType = 0.1 * u.deg
    max_region_number: int = 10000
    binsz: AngleType = 0.05 * u.deg


class WobbleRegionsFinderConfig(BaseConfig):
    """
    Config section for getting main information for creating a
    WobbleRegionsFinder object.
    """

    n_off_regions: int = 1
    binsz: AngleType = 0.05 * u.deg


class RegionsConfig(BaseConfig):
    """
    Config section for getting main information for creating a Regions object.
    """

    type: str = ""
    name: str = ""
    position: SkyPositionConfig = SkyPositionConfig()
    parameters: dict = {}


class ExclusionRegionsConfig(BaseConfig):
    """
    Config section for getting information on exclusion regions for creating
    an Exclusion Mask object.
    """

    target_source: bool = True
    regions: List[RegionsConfig] = []


class SafeMaskMethodsEnum(str, Enum):
    """
    Config section for list of methods for creating Safe Mask Reduction Makers
    object.
    """

    aeff_default = "aeff-default"
    aeff_max = "aeff-max"
    edisp_bias = "edisp-bias"
    offset_max = "offset-max"
    bkg_peak = "bkg-peak"
    custom_mask = "custom-mask"


class MapSelectionEnum(str, Enum):
    """
    Config section for list of methods for creating a Dataset Maker object.
    """

    counts = "counts"
    exposure = "exposure"
    background = "background"
    psf = "psf"
    edisp = "edisp"


# Dataset Reduction Makers config
class BackgroundConfig(BaseConfig):
    """
    Config section for getting main information for creating a Background
    Reduction Makers object.
    """

    method: BackgroundMethodEnum = BackgroundMethodEnum.reflected
    region_finder_method: BackgroundRegionFinderMethodEnum = BackgroundRegionFinderMethodEnum.wobble
    parameters: dict = {}
    exclusion: ExclusionRegionsConfig = ExclusionRegionsConfig()


class SafeMaskConfig(BaseConfig):
    """
    Config section for getting main information for creating a Safe Mask Makers
    object.
    """

    methods: List[SafeMaskMethodsEnum] = []
    parameters: dict = {}


def get_filtered_observations(dl3_path, obs_config, log):
    """
    From the path of the DL3 index files, create gammapy Observations object and
    apply any observation filters provided in the obs_config object to return
    the selected Observations object.

    Parameters
    ----------
    dl3_path: `pathlib.Path`
        Path to the DL3 index files, to create `gammapy.data.DataStore` object.
    obs_config: `asgardpy.base.reduction.ObservationsConfig`
        Config information for creating the `gammapy.data.Observations` object.
    log: `logging()`
        Common log file.

    Return
    ------
    observations: `gammapy.data.Observations`
        Selected list of Observation object
    """
    datastore = DataStore.from_dir(dl3_path)

    obs_time = obs_config.obs_time
    obs_list = obs_config.obs_ids
    obs_cone = obs_config.obs_cone

    # In case the obs_table is not sorted.
    obs_table = datastore.obs_table.group_by("OBS_ID")

    # Use the given list of Observation IDs to select Observations
    if len(obs_list) > 0:
        # if len(obs_list) > 2:
        # list of observation ids to be included
        filtered_obs_ids = obs_list
        # else:  # Find another way to make the distinction between list and range
        # the list has a min and max value to use this method
        #    id_select = {
        #        "type": "par_box",
        #        "variable": "OBS_ID",
        #        "value_range": obs_list,
        #    }
        #    obs_table = obs_table.select_observations(id_select)

    # Filter the Observations using the Time interval range provided
    if obs_time.intervals[0].start != Time("0", format="mjd"):
        t_start = Time(obs_time.intervals[0].start, format=obs_time.format)
        t_stop = Time(obs_time.intervals[0].stop, format=obs_time.format)

        gti_select = {
            "type": "time_box",
            "time_range": [t_start, t_stop],
        }
        obs_table = obs_table.select_observations(gti_select)

    # For 3D Dataset, use a sky region to select Observations
    if obs_cone.lon != 0 * u.deg:
        cone_select = {
            "type": "sky_circle",
            "frame": obs_cone.frame,
            "lon": obs_cone.lon,
            "lat": obs_cone.lat,
            "radius": obs_cone.radius,
            "border": 0 * u.deg,  # Default?
        }
        obs_table = obs_table.select_observations(cone_select)

    if filtered_obs_ids:
        filtered_obs_ids = np.intersect1d(filtered_obs_ids, obs_table["OBS_ID"].data)
    else:
        filtered_obs_ids = obs_table["OBS_ID"].data

    obs_ids_str = " ".join(map(str, filtered_obs_ids))
    log.info("Observation ID list selected: %s", obs_ids_str)

    # IRFs selection
    irfs_selected = obs_config.required_irfs
    observations = datastore.get_observations(filtered_obs_ids, required_irf=irfs_selected)

    return observations


def get_dataset_reference(tag, geom, geom_config, name=None):
    """
    Create a base Dataset object to fill up with the appropriate {1, 3}D type
    of DL3 data to generate the reduced DL4 dataset, using the given base
    geometry and relevant axes details.

    Parameters
    ----------
    tag: str
        Determining either the {1, 3}d Dataset type.
    geom: 'gammapy.maps.RegionGeom' or `gammapy.maps.WcsGeom`
        Appropriate Base geometry objects for {1, 3}D type of DL4 Datasets.
    geom_config: `asgardpy.base.geom.GeomConfig`
        Config information on creating the Base Geometry of the DL4 dataset.
    name: str
        Name for the dataset.

    Return
    ------
    dataset_reference: `gammapy.dataset.SpectrumDataset` or
        `gammapy.dataset.MapDataset`
        Appropriate Dataset reference for {1, 3}D type of DL4 Datasets.
    """
    dataset_reference = None

    if tag == "1d":
        for axes_ in geom_config.axes:
            if axes_.name == "energy_true":
                energy_axis = get_energy_axis(axes_)
                dataset_reference = SpectrumDataset.create(
                    geom=geom,
                    name=name,
                    energy_axis_true=energy_axis,
                )
    else:  # For tag == "3d"
        binsize_irf = geom_config.wcs.binsize_irf.to_value("deg")
        dataset_reference = MapDataset.create(
            geom=geom,
            name=name,
            binsz_irf=binsize_irf,
        )

    return dataset_reference


def get_dataset_maker(tag, dataset_config):
    """
    Create a Dataset Maker object to support creating an appropriate {1, 3}D
    type of DL4 Dataset along with other reduction makers.

    Parameters
    ----------
    tag: str
        Determining either the {1, 3}d Dataset type.
    dataset_config: `asgardpy.data.dataset_1d.Dataset1DInfoConfig` or
        `asgardpy.data.dataset_3d.Dataset3DInfoConfig`
        Config information on creating appropriate {1, 3}d DL4 dataset type.

    Return
    ------
    dataset_maker: `gammapy.makers.SpectrumDatasetMaker` or
        `gammapy.makers.MapDatasetMaker`
        Appropriate Dataset Maker for {1, 3}D type of DL4 Datasets.
    """
    if tag == "1d":
        dataset_maker = SpectrumDatasetMaker(
            containment_correction=dataset_config.containment_correction,
            selection=dataset_config.map_selection,
        )
    else:  # for tag == "3d"
        dataset_maker = MapDatasetMaker(selection=dataset_config.map_selection)

    return dataset_maker


def get_safe_mask_maker(safe_config):
    """
    Generate Safe mask reduction maker as per the given config information.

    Parameters
    ---------
    safe_config: `asgardpy.base.reduction.SafeMaskConfig`
        Config information to create `gammapy.makers.SafeMaskMaker` object.

    Return
    ------
    safe_maker: `gammapy.makers.SafeMaskMaker`
        Gammapy Dataset Reduction Maker, for safe data range mask.
    """
    pars = safe_config.parameters

    if len(safe_config.methods) != 0:
        if "custom-mask" not in safe_config.methods:
            safe_maker = SafeMaskMaker(methods=safe_config.methods, **pars)
        else:
            safe_maker = None
    else:
        safe_maker = None

    return safe_maker


def get_exclusion_region_mask(
    exclusion_params,
    excluded_geom,
    exclusion_regions,
    config_target,
    geom_config,
    log,
):
    """
    Generate from a given parameters, base geometry for exclusion mask, list
    of exclusion regions, config information on the target source and the base
    geometry for the exclusion mask, a background exclusion region mask.

    Parameters
    ----------
    exclusion_params: `asgardpy.base.reduction.ExclusionRegionsConfig`
        Config information on the list of Exclusion Regions
    excluded_geom: 'gammapy.maps.RegionGeom' or `gammapy.maps.WcsGeom`
        Appropriate Base geometry objects for exclusion regions for {1, 3}D
        type of DL4 Datasets.
    exclusion_regions: list of `gammapy.maps.WcsMap`
        Existing list of excluded regions.
    config_target: `asgardpy.config.generator.AsgardpyConfig.target`
        Config information on the target source
    geom_config: `asgardpy.base.geom.GeomConfig`
        Config information on creating the Base Geometry of the DL4 dataset.
    log: `logging()`
        Common log file.
    Return
    ------
    exclusion_mask: `gammapy.maps.WcsNDMap`
        Boolean region mask for the exclusion regions
    """
    if len(exclusion_params.regions) != 0:
        # Fetch information from config
        for region in exclusion_params.regions:
            if region.name == "":
                # Using the sky position information without the source name.
                coord = region.position
                center_ex = SkyCoord(
                    u.Quantity(coord.lon), u.Quantity(coord.lat), frame=coord.frame
                ).icrs
            else:
                # Using Sesame name resolver
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
                log.error(f"Unknown type of region passed {region.type}")
            exclusion_regions.append(excluded_region)

    # Check if a catalog data is given with exclusion radius
    if config_target.use_catalog.exclusion_radius != 0 * u.deg:
        catalog = CATALOG_REGISTRY.get_cls(config_target.use_catalog.name)()

        # Only use source positions from the Catalog within the base geometry
        inside_geom = excluded_geom.to_image().contains(catalog.positions)

        idx_list = np.nonzero(inside_geom)[0]
        for i in idx_list:
            exclusion_regions.append(
                CircleSkyRegion(
                    center=catalog[i].position,
                    radius=config_target.use_catalog.exclusion_radius,
                )
            )

    # Apply the exclusion regions mask on the base geometry to get the final
    # boolean mask
    if len(exclusion_regions) > 0:
        exclusion_mask = ~excluded_geom.region_mask(exclusion_regions)
    else:
        exclusion_mask = None

    return exclusion_mask


def get_bkg_maker(bkg_config, exclusion_mask):
    """
    Generate Background reduction maker by including a boolean exclusion mask,
    with methods to find background normalization using the information
    provided in the config for using specific methods.

    Parameters
    ----------
    bkg_config: `asgardpy.base.reduction.BackgroundConfig`
        Config information for evaluating a particular Background
        normalization maker for dataset reduction.
    exclusion_mask: `gammapy.maps.WcsNDMap`
        Boolean region mask for the exclusion regions

    Return
    ------
    bkg_maker: `gammapy.makers.background()`
        Appropriate gammapy Background Maker objects as per the config.
    """
    if bkg_config.method == "reflected":
        if bkg_config.region_finder_method == "wobble":
            region_finder = WobbleRegionsFinder(**bkg_config.parameters)
        elif bkg_config.region_finder_method == "reflected":
            region_finder = ReflectedRegionsFinder(**bkg_config.parameters)

        bkg_maker = ReflectedRegionsBackgroundMaker(
            region_finder=region_finder, exclusion_mask=exclusion_mask
        )
    elif bkg_config.method == "fov_background":
        bkg_maker = FoVBackgroundMaker(exclusion_mask=exclusion_mask, **bkg_config.parameters)
    elif bkg_config.method == "ring":
        bkg_maker = RingBackgroundMaker(exclusion_mask=exclusion_mask, **bkg_config.parameters)
    else:
        bkg_maker = None

    return bkg_maker


def generate_dl4_dataset(
    tag,
    observations,
    dataset_reference,
    dataset_maker,
    bkg_maker,
    safe_maker,
    n_jobs,
    parallel_backend,
):
    """
    From the given Observations, Dataset reference and various Makers,
    use the multiprocessing method with DatasetsMaker, create the appropriate
    DL4 Dataset for {1, 3}D type of DL3 data.

    Parameters
    ----------
    tag: str
        Determining either the {1, 3}d Dataset type.
    observations: `gammapy.data.Observations`
        Selected list of Observation object
    dataset_reference: `gammapy.dataset.SpectrumDataset` or
        `gammapy.dataset.MapDataset`
        Appropriate Dataset reference for {1, 3}D type of DL4 Datasets.
    dataset_maker: `gammapy.makers.MapDatasetMaker` or
        `gammapy.makers.SpectrumDatasetMaker`
        Appropriate gammapy object to bin the Observations Map data or 1D
        spectrum extraction data for {1, 3}D type of DL4 Datasets.
    bkg_maker: `gammapy.makers.background()`
        Appropriate gammapy Background Maker objects as per the config.
    safe_maker: `gammapy.makers.SafeMaskMaker`
        Gammapy Dataset Reduction Maker, for safe data range mask.
    n_jobs: int
        Number of parallel processing jobs for `gammapy.makers.DatasetsMaker`
    parallel_backend: str
        Name of the parallel backend used for the parallel processing. By
        default "multiprocessing" is used for now.

    Return
    ------
    datasets: `gammapy.datasets.Datasets`
        A Datasets object containing appropriate `gammapy.dataset.MapDataset`
        or `gammapy.dataset.SpectrumDataset` for {1, 3}D type of DL3 dataset.
    """
    if safe_maker:
        makers = [dataset_maker, safe_maker, bkg_maker]
    else:
        makers = [dataset_maker, bkg_maker]

    if tag == "1d":
        datasets_maker = DatasetsMaker(
            makers,
            stack_datasets=False,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )
    else:
        datasets_maker = DatasetsMaker(
            makers,
            stack_datasets=False,  # Add to the config for 3D dataset?
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            cutout_mode="trim",  # Add to the config for 3D dataset?
            # cutout_width=2*offset_max  # As used in the API, from geom.selection
        )

    datasets = datasets_maker.run(dataset_reference, observations)

    return datasets
