"""
Classes containing the Base for the Analysis steps and some Basic Config types.
"""

from enum import Enum
from pathlib import Path
from typing import Annotated, List, Union

from astropy import units as u
from astropy.time import Time
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    PlainSerializer,
    WithJsonSchema,
)

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

# Following suggested answers from
# https://stackoverflow.com/questions/76686888/using-bson-objectid-in-pydantic-v2/


# Basic Quantities Type for building the Config
def validate_angle_type(v: str) -> u.Quantity:
    """Validation for Base Angle Type Quantity"""
    if isinstance(v, u.Quantity):
        v_ = v
    elif isinstance(v, str):
        v_ = u.Quantity(v)
    if v_.unit.physical_type != "angle":
        raise ValueError(f"Invalid unit for angle: {v_.unit!r}")
    else:
        return v_


# Base Angle Type Quantity
AngleType = Annotated[
    Union[str, u.Quantity],
    AfterValidator(validate_angle_type),
    PlainSerializer(lambda x: u.Quantity(x), return_type=u.Quantity),
    WithJsonSchema({"type": "string"}, mode="serialization"),
]


def validate_energy_type(v: str) -> u.Quantity:
    """Validation for Base Energy Type Quantity"""
    if isinstance(v, u.Quantity):
        v_ = v
    elif isinstance(v, str):
        v_ = u.Quantity(v)
    if v_.unit.physical_type != "energy":
        raise ValueError(f"Invalid unit for energy: {v_.unit!r}")
    else:
        return v_


# Base Energy Type Quantity
EnergyType = Annotated[
    Union[str, u.Quantity],
    AfterValidator(validate_energy_type),
    PlainSerializer(lambda x: u.Quantity(x), return_type=u.Quantity),
    WithJsonSchema({"type": "string"}, mode="serialization"),
]


def validate_time_type(v: str) -> Time:
    """Validation for Base Time Type Quantity"""
    if isinstance(v, Time):
        v_ = v
    elif isinstance(v, str):
        v_ = Time(v)
    return v_


# Base Time Type Quantity
TimeType = Annotated[
    Union[str, Time],
    AfterValidator(validate_time_type),
    PlainSerializer(lambda x: Time(x), return_type=Time),
    WithJsonSchema({"type": "string"}, mode="serialization"),
]


def validate_path_type(v: str) -> Path:
    """Validation for Base Path Type Quantity"""
    if isinstance(v, Path):
        v_ = v
    elif isinstance(v, str):
        if v == "None":
            v_ = Path(".")
        else:
            v_ = Path(v)

    # Only check if the file location or directory path exists
    path_ = v_.resolve()
    if path_.is_file():
        path_ = path_.parent
    if not path_.exists():
        raise ValueError(f"Path {v_} does not exist")
    else:
        return v_.resolve()


# Base Path Type Quantity
PathType = Annotated[
    Union[str, Path],
    AfterValidator(validate_path_type),
    PlainSerializer(lambda x: Path(x), return_type=Path),
    WithJsonSchema({"type": "string"}, mode="serialization"),
]


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

    model_config = ConfigDict(
        validate_default=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )


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
