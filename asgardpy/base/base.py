"""
Classes containing the Base for the Analysis steps and some Basic Config types.
"""

from enum import Enum
from pathlib import Path
from typing import List

from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time
from pydantic import BaseModel

__all__ = [
    "AngleType",
    "BaseConfig",
    "EnergyRangeConfig",
    "EnergyType",
    "FrameEnum",
    "PathType",
    "TimeFormatEnum",
    "TimeIntervalsConfig",
    "TimeRangeConfig",
    "TimeType",
]


# Basic Quantities Type for building the Config
class AngleType(Angle):
    """Base Angle Type Quantity"""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return Angle(v)


class EnergyType(u.Quantity):
    """Base Energy Type Quantity"""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        v = u.Quantity(v)
        if v.unit.physical_type != "energy":
            raise ValueError(f"Invalid unit for energy: {v.unit!r}")
        return v


class TimeType(Time):
    """Base Time Type Quantity"""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return Time(v)


class PathType(str):
    """Base Path Type Quantity"""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if v == "None":
            return Path(".")
        else:
            path_ = Path(v).resolve()
            # Only check if the file location or directory path exists
            if path_.is_file():
                path_ = path_.parent

            if path_.exists():
                return Path(v)
            else:
                raise ValueError(f"Path {v} does not exist")


class FrameEnum(str, Enum):
    """Config section for list of frames on creating a SkyCoord object."""

    icrs = "icrs"
    galactic = "galactic"


class TimeFormatEnum(str, Enum):
    """Config section for list of formats for creating a Time object."""

    datetime = "datetime"
    fits = "fits"
    iso = "iso"
    isot = "isot"
    jd = "jd"
    mjd = "mjd"
    unix = "unix"


class BaseConfig(BaseModel):
    """
    Base Config class for creating other Config sections with specific encoders.
    """

    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        json_encoders = {
            Angle: lambda v: f"{v.value} {v.unit}",
            u.Quantity: lambda v: f"{v.value} {v.unit}",
            Time: lambda v: f"{v.value}",
            Path: lambda v: PathType(v),
        }


# Basic Quantity ranges Type for building the Config
class TimeRangeConfig(BaseConfig):
    """
    Config section for getting a time range information for creating a Time
    object.
    """

    start: TimeType = Time("0", format="mjd")
    stop: TimeType = Time("0", format="mjd")


class TimeIntervalsConfig(BaseConfig):
    """
    Config section for getting main information for creating a Time Intervals
    object.
    """

    format: TimeFormatEnum = TimeFormatEnum.iso
    intervals: List[TimeRangeConfig] = [TimeRangeConfig()]


class EnergyRangeConfig(BaseConfig):
    """
    Config section for getting a energy range information for creating an
    Energy type Quantity object.
    """

    min: EnergyType = 1 * u.GeV
    max: EnergyType = 1 * u.TeV
