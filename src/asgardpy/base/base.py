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
    "TimeIntervalsType",
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


class TimeIntervalsType(List):
    """
    Config section for getting main information for creating a Time Intervals
    object.
    """

    intervals: dict = {"format": TimeFormatEnum.iso, "start": "1970-01-01", "stop": "2000-01-01"}

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if Time(v["start"], format=v["format"]):
            v["start"] = Time(v["start"], format=v["format"])
        else:
            raise ValueError(f"{v['start']} is not the right Time value for format {v['format']}")

        if Time(v["stop"], format=v["format"]):
            v["stop"] = Time(v["stop"], format=v["format"])
        else:
            raise ValueError(f"{v['stop']} is not the right Time value for format {v['format']}")

        return v


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
            Path: lambda v: PathType(v),
            Time: lambda v: Time(v).iso,
        }


# Basic Quantity ranges Type for building the Config
class EnergyRangeConfig(BaseConfig):
    """
    Config section for getting a energy range information for creating an
    Energy type Quantity object.
    """

    min: EnergyType = 1 * u.GeV
    max: EnergyType = 1 * u.TeV
