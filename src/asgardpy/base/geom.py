"""
Classes containing the Geometry config parameters for the high-level interface
and also some functions to support creating the base Geometry of various
DL4 datasets.
"""

import re
from enum import Enum

from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, RegionGeom, WcsGeom
from regions import CircleSkyRegion, PointSkyRegion

from asgardpy.base.base import AngleType, BaseConfig, EnergyType, FrameEnum

__all__ = [
    "EnergyAxisConfig",
    "EnergyEdgesCustomConfig",
    "GeomConfig",
    "MapAxesConfig",
    "MapFrameShapeConfig",
    "ProjectionEnum",
    "SelectionConfig",
    "SkyPositionConfig",
    "WcsConfig",
    "create_counts_map",
    "generate_geom",
    "get_energy_axis",
    "get_source_position",
]


# Basic Components to define the main GeomConfig
class EnergyAxisConfig(BaseConfig):
    """
    Config section for getting main information for creating an Energy type
    MapAxis object.
    """

    min: EnergyType = 1 * u.GeV
    max: EnergyType = 1 * u.TeV
    nbins: int = 5
    per_decade: bool = True


class EnergyEdgesCustomConfig(BaseConfig):
    """
    Config section for getting information of energy edges for creating an
    Energy type MapAxis object.
    """

    edges: list[float] = []
    unit: str = "TeV"


class MapAxesConfig(BaseConfig):
    """
    Config section for getting main information for creating a MapAxis object.
    """

    name: str = "energy"
    axis: EnergyAxisConfig = EnergyAxisConfig()
    axis_custom: EnergyEdgesCustomConfig = EnergyEdgesCustomConfig()


class SelectionConfig(BaseConfig):
    """
    Config section for extra selection criteria on creating a MapDataset object.
    """

    offset_max: AngleType = 2.5 * u.deg


class MapFrameShapeConfig(BaseConfig):
    """
    Config section for getting frame size information for creating a Map object.
    """

    width: AngleType = 5 * u.deg
    height: AngleType = 5 * u.deg


class SkyPositionConfig(BaseConfig):
    """
    Config section for getting main information for creating a SkyCoord object.
    """

    frame: FrameEnum = FrameEnum.icrs
    lon: AngleType = 0 * u.deg
    lat: AngleType = 0 * u.deg
    radius: AngleType = 0 * u.deg


class ProjectionEnum(str, Enum):
    """Config section for list of sky projection methods."""

    tan = "TAN"
    car = "CAR"


class WcsConfig(BaseConfig):
    """
    Config section for getting main sky projection information for creating a
    base Geometry.
    """

    skydir: SkyPositionConfig = SkyPositionConfig()
    binsize: AngleType = 0.1 * u.deg
    proj: ProjectionEnum = ProjectionEnum.tan
    map_frame_shape: MapFrameShapeConfig = MapFrameShapeConfig()
    binsize_irf: AngleType = 0.2 * u.deg


class GeomConfig(BaseConfig):
    """
    Config section for getting main information for creating a base Geometry.
    """

    wcs: WcsConfig = WcsConfig()
    selection: SelectionConfig = SelectionConfig()
    axes: list[MapAxesConfig] = [MapAxesConfig()]
    from_events_file: bool = True
    reco_psf: bool = False


def get_energy_axis(axes_config, only_edges=False, custom_range=False):
    """
    Read from a MapAxesConfig section to return the required energy axis/range.

    Parameters
    ----------
    axes_config: `asgardpy.base.geom.MapAxesConfig`
        COnfig for generating an energy axis
    only_edges: bool
        If True, only return an array of energy edges
    custom_range: bool
        If True, use the given list of energy edges and energy unit.

    Return
    ------
    energy_range: `gammapy.maps.MapAxis`
        Energy Axis formed from the given config parameters
    """
    energy_axis = axes_config.axis
    energy_axis_custom = axes_config.axis_custom

    if custom_range:
        energy_range = energy_axis_custom.edges * u.Unit(energy_axis_custom.unit)
        # print("Energy edges evaluated from the geom config info", energy_range)
    else:
        energy_range = MapAxis.from_energy_bounds(
            energy_min=u.Quantity(energy_axis.min),
            energy_max=u.Quantity(energy_axis.max),
            nbin=int(energy_axis.nbins),
            per_decade=energy_axis.per_decade,
            name=axes_config.name,
        )
        if only_edges:
            energy_range = energy_range.edges

    return energy_range


def get_source_position(target_region, fits_header=None):
    """
    Function to fetch the target source position and the angular radius for
    the generating the Counts Map or ON region.

    Parameters
    ----------
    target_region: `asgardpy.data.geom.SkyPositionConfig`
        Config containing the information on the target source position
    fits_header: `astropy.io.fits.Header`
        FITS Header information of the events fits file, only for Fermi-LAT
        DL3 files. If None is passed, the information is collected from the
        config_target.

    Return
    ------
    center_pos: dict
        Dict information on the central `astropy.coordinates.SkyCoord` position
        and `astropy.units.Quantity` angular radius.
    """
    if fits_header:
        try:
            dsval2 = fits_header["DSVAL2"]
            list_str_check = re.findall(r"[-+]?\d*\.\d+|\d+", dsval2)
        except KeyError:
            history = str(fits_header["HISTORY"])
            str_ = history.split("angsep(RA,DEC,")[1]
            list_str_check = re.findall(r"[-+]?\d*\.\d+|\d+", str_)[:3]
        ra_pos, dec_pos, events_radius = (float(k) for k in list_str_check)

        source_pos = SkyCoord(ra_pos, dec_pos, unit="deg", frame="fk5")
    else:
        source_pos = SkyCoord(
            u.Quantity(target_region.lon),
            u.Quantity(target_region.lat),
            frame=target_region.frame,
        )
        events_radius = target_region.radius

    center_pos = {"center": source_pos, "radius": events_radius}

    return center_pos


def create_counts_map(geom_config, center_pos):
    """
    Generate the counts Map object using the information provided in the
    geom section of the Config and the dict information on the position and
    Counts Map size of the target source. Used currently only for Fermi-LAT
    DL3 files.

    Parameters
    ----------
    geom_config: `asgardpy.base.geom.GeomConfig`
        Config information on creating the Base Geometry of the DL4 dataset.
    center_pos: dict
        Dict information on the central `astropy.coordinates.SkyCoord` position
        and `astropy.units.Quantity` angular radius.

    Return
    ------
    counts_map: `gammapy.maps.Map`
        Map object of the Counts information.
    """
    energy_axes = geom_config.axes[0]
    energy_axis = get_energy_axis(energy_axes)
    bin_size = geom_config.wcs.binsize.to_value(u.deg)

    # For fetching information from the events fits file to resize the Map size.
    if geom_config.from_events_file:
        counts_map = Map.create(
            skydir=center_pos["center"].galactic,
            binsz=bin_size,
            npix=(
                int(center_pos["radius"] * 2 / bin_size),
                int(center_pos["radius"] * 2 / bin_size),
            ),  # Using the limits from the events fits file
            proj=geom_config.wcs.proj,
            frame="galactic",
            axes=[energy_axis],
            dtype=float,
        )
    else:
        # Using the config information
        width_ = geom_config.wcs.map_frame_shape.width.to_value(u.deg)
        width_in_pixel = int(width_ / bin_size)
        height_ = geom_config.wcs.map_frame_shape.height.to_value(u.deg)
        height_in_pixel = int(height_ / bin_size)

        counts_map = Map.create(
            skydir=center_pos["center"].galactic,
            binsz=bin_size,
            npix=(width_in_pixel, height_in_pixel),
            proj=geom_config.wcs.proj,
            frame="galactic",
            axes=[energy_axis],
            dtype=float,
        )

    return counts_map


def generate_geom(tag, geom_config, center_pos):
    """
    Generate from a given target source position, the geometry of the ON
    events and the axes information on reco energy and true energy,
    for which a DL4 dataset of a given tag, can be defined.

    Can also generate Base geometry for the excluded regions for a 3D dataset.

    Parameters
    ----------
    tag: str
        Determining either the {1, 3}d Dataset type of "excluded", for creating
        base geometry for the excluded regions.
    geom_config: `asgardpy.base.geom.GeomConfig`
        Config information on creating the Base Geometry of the DL4 dataset.
    center_pos: dict
        Dict information on the central `astropy.coordinates.SkyCoord` position
        and `astropy.units.Quantity` angular radius.

    Return
    ------
    geom: 'gammapy.maps.RegionGeom' or `gammapy.maps.WcsGeom`
        appropriate Base geometry objects for {1, 3}D type of DL4 Datasets.
    """
    # Getting the energy axes
    axes_list = geom_config.axes

    for axes_ in axes_list:
        if axes_.name == "energy":
            custom_range = len(axes_.axis_custom.edges) > 1
            energy_axis = get_energy_axis(axes_, custom_range=custom_range)

            if custom_range:
                energy_axis = MapAxis.from_energy_edges(energy_axis, name=axes_.name)

    if tag == "1d":
        # Defining the ON region's geometry for DL4 dataset
        if center_pos["radius"] == 0 * u.deg:
            on_region = PointSkyRegion(center_pos["center"])
            # Hack to allow for the joint fit
            # (otherwise pointskyregion.contains returns None)
            on_region.meta = {"include": False}
        else:
            on_region = CircleSkyRegion(
                center=center_pos["center"],
                radius=u.Quantity(center_pos["radius"]),
            )

        geom = RegionGeom.create(region=on_region, axes=[energy_axis])

    else:
        width_ = geom_config.wcs.map_frame_shape.width.to_value("deg")
        height_ = geom_config.wcs.map_frame_shape.height.to_value("deg")

        geom_params = {}

        if "ex" in tag:  # For exclusion regions - include for 1D data as well
            bin_size = geom_config.wcs.binsize.to_value("deg")
            width_ = int(width_ / bin_size)
            height_ = int(height_ / bin_size)

            if "1d" in tag:
                geom_params["npix"] = (width_, height_)
            else:
                # 3D-ex
                geom_params["width"] = (width_, height_)
        else:
            # 3D dataset for DL4 creation
            # print("Energy edges before creating the geom", energy_axis.name)
            geom_params["width"] = (width_, height_)
            geom_params["axes"] = [energy_axis]

        geom_params["skydir"] = center_pos["center"]
        geom_params["frame"] = center_pos["center"].frame
        geom_params["binsz"] = geom_config.wcs.binsize
        geom_params["proj"] = geom_config.wcs.proj

        # Main geom for 3D Dataset
        geom = WcsGeom.create(**geom_params)

    return geom
