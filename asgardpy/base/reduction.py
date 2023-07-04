"""
Classes containing the Dataset Reduction config parameters for the high-level
interface.
"""

from enum import Enum
from typing import List

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from gammapy.data import DataStore
from gammapy.datasets import MapDataset, SpectrumDataset
from gammapy.makers import (
    DatasetsMaker,
    FoVBackgroundMaker,
    ReflectedRegionsBackgroundMaker,
    ReflectedRegionsFinder,
    RingBackgroundMaker,
    SafeMaskMaker,
    WobbleRegionsFinder,
)
from regions import CircleAnnulusSkyRegion, CircleSkyRegion

from asgardpy.base import (
    AngleType,
    BaseConfig,
    PathType,
    SkyPositionConfig,
    TimeIntervalsConfig,
)
from asgardpy.base.geom import generate_geom, get_energy_axis

__all__ = [
    "BackgroundConfig",
    "BackgroundMethodEnum",
    "BackgroundRegionFinderMethodEnum",
    "ExclusionRegionsConfig",
    "generate_dl4_dataset",
    "get_bkg_maker",
    "get_dataset_template",
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
    spectrum = "1d"
    cube = "3d"


class RequiredHDUEnum(str, Enum):
    aeff = "aeff"
    bkg = "bkg"
    edisp = "edisp"
    psf = "psf"
    rad_max = "rad_max"
    point_like = "point-like"
    full_enclosure = "full-enclosure"


class ObservationsConfig(BaseConfig):
    obs_ids: List[int] = []
    obs_file: PathType = PathType(".")
    obs_time: TimeIntervalsConfig = TimeIntervalsConfig()
    obs_cone: SkyPositionConfig = SkyPositionConfig()
    required_irfs: List[RequiredHDUEnum] = [RequiredHDUEnum.aeff]


class BackgroundMethodEnum(str, Enum):
    reflected = "reflected"
    fov = "fov_background"
    ring = "ring"


class BackgroundRegionFinderMethodEnum(str, Enum):
    reflected = "reflected"
    wobble = "wobble"


class ReflectedRegionFinderConfig(BaseConfig):
    angle_increment: AngleType = 0.01 * u.deg
    min_distance: AngleType = 0.1 * u.deg
    min_distance_input: AngleType = 0.1 * u.deg
    max_region_number: int = 10000
    binsz: AngleType = 0.05 * u.deg


class WobbleRegionsFinderConfig(BaseConfig):
    n_off_regions: int = 1
    binsz: AngleType = 0.05 * u.deg


class RegionsConfig(BaseConfig):
    type: str = ""
    name: str = ""
    position: SkyPositionConfig = SkyPositionConfig()
    parameters: dict = {}


class ExclusionRegionsConfig(BaseConfig):
    target_source: bool = True
    regions: List[RegionsConfig] = []


class SafeMaskMethodsEnum(str, Enum):
    aeff_default = "aeff-default"
    aeff_max = "aeff-max"
    edisp_bias = "edisp-bias"
    offset_max = "offset-max"
    bkg_peak = "bkg-peak"
    custom_mask = "custom-mask"


class MapSelectionEnum(str, Enum):
    counts = "counts"
    exposure = "exposure"
    background = "background"
    psf = "psf"
    edisp = "edisp"


# Dataset Reduction Makers config
class BackgroundConfig(BaseConfig):
    method: BackgroundMethodEnum = BackgroundMethodEnum.reflected
    region_finder_method: BackgroundRegionFinderMethodEnum = BackgroundRegionFinderMethodEnum.wobble
    parameters: dict = {}
    exclusion: ExclusionRegionsConfig = ExclusionRegionsConfig()


class SafeMaskConfig(BaseConfig):
    methods: List[SafeMaskMethodsEnum] = []
    parameters: dict = {}


def get_filtered_observations(dl3_path, obs_config, log):
    """
    From the DataStore object, apply any observation filters provided in
    the obs_config object to create and return an Observations object.
    """
    # Could be generalized along with the same in dataset_3d
    datastore = DataStore.from_dir(dl3_path)

    obs_time = obs_config.obs_time
    obs_list = obs_config.obs_ids
    obs_cone = obs_config.obs_cone

    obs_table = datastore.obs_table.group_by("OBS_ID")

    if len(obs_list) > 0:
        id_select = {
            "type": "par_box",
            "variable": "OBS_ID",
            "value_range": obs_list,
        }
        obs_table = obs_table.select_observations(id_select)
    # Filter using the Time interval range provided
    if obs_time.intervals[0].start != Time("0", format="mjd"):
        t_start = Time(obs_time.intervals[0].start, format=obs_time.format)
        t_stop = Time(obs_time.intervals[0].stop, format=obs_time.format)

        gti_select = {
            "type": "time_box",
            "time_range": [t_start, t_stop],
        }
        obs_table = obs_table.select_observations(gti_select)

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

    filtered_obs_ids = obs_table["OBS_ID"].data
    log.info(f"Observation ID list selected: {filtered_obs_ids}")

    # IRFs selection
    irfs_selected = obs_config.required_irfs
    observations = datastore.get_observations(filtered_obs_ids, required_irf=irfs_selected)

    return observations


def get_dataset_template(tag, geom, geom_config, name=None):
    """
    Common for nD dataset of gadf-dl3 type
    """
    dataset_template = None

    if tag == "1d":
        for axes_ in geom_config.axes:
            if axes_.name == "energy_true":
                energy_axis = get_energy_axis(axes_)
                dataset_template = SpectrumDataset.create(
                    geom=geom,
                    name=name,
                    energy_axis_true=energy_axis,
                )
    else:  # For tag == "3d"
        binsize_irf = geom_config.wcs.binsize_irf.to_value("deg")
        dataset_template = MapDataset.create(
            geom=geom,
            name=name,
            binsz_irf=binsize_irf,
        )

    return dataset_template


def get_safe_mask_maker(safe_config):
    """
    Generate Safe mask reduction maker as per the selections provided in
    the config.
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


def get_bkg_maker(bkg_config, geom_config, exclusion_regions, config_target, log):
    """
    Generate Background reduction maker by including an Exclusion mask
    with any exclusion regions' information on a Map geometry using the
    information provided in the config.
    """
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
                log.error(f"Unknown type of region passed {region.type}")
            exclusion_regions.append(excluded_region)
    else:
        center_ex = SkyCoord(
            u.Quantity(config_target.sky_position.lon),
            u.Quantity(config_target.sky_position.lat),
            frame=config_target.sky_position.frame,
        )
        exclusion_regions = []

    excluded_geom = generate_geom(
        tag="excluded",
        geom_config=geom_config,
        center_pos={"center": center_ex},
    )

    if len(exclusion_regions) > 0:
        exclusion_mask = ~excluded_geom.region_mask(exclusion_regions)
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
    elif bkg_config.method == "fov_background":
        bkg_maker = FoVBackgroundMaker(**bkg_config.parameters)
    elif bkg_config.method == "ring":
        bkg_maker = RingBackgroundMaker(**bkg_config.parameters)
    else:
        bkg_maker = None

    return bkg_maker


def generate_dl4_dataset(
    tag,
    observations,
    dataset_template,
    dataset_maker,
    bkg_maker,
    safe_maker,
    n_jobs,
    parallel_backend,
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

    datasets = datasets_maker.run(dataset_template, observations)

    return datasets
