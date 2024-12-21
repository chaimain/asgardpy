"""
Classes containing the Base for the Analysis steps and some Basic Config types.
"""

import html
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

from astropy import units as u
from astropy.time import Time
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    GetCoreSchemaHandler,
    PlainSerializer,
)
from pydantic_core import core_schema

__all__ = [
    "AngleType",
    "BaseConfig",
    "EnergyRangeConfig",
    "EnergyType",
    "FrameEnum",
    "PathType",
    "TimeFormatEnum",
    "TimeInterval",
]


# Basic Quantities Type for building the Config
"""
class JsonQuantityEncoder(json.JSONEncoder):
    #Support for quantities that JSON default encoder

    def default(self, obj):
        if isinstance(obj, u.Quantity):
            return obj.to_string()

        return json.JSONEncoder.default(self, obj)


# Todo: replace by QuantityType and pydantic TypeAdapter
class JsonQuantityDecoder(json.JSONDecoder):
    #Support for quantities that JSON default encoder

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(data):
        for key, value in data.items():
            try:
                data[key] = u.Quantity(value)
            except TypeError:
                continue
        return data
"""


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
    str | u.Quantity,
    BeforeValidator(validate_angle_type),
    PlainSerializer(lambda x: u.Quantity(x), when_used="json-unless-none", return_type=u.Quantity),
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
    str | u.Quantity,
    BeforeValidator(validate_energy_type),
    PlainSerializer(lambda x: u.Quantity(x), when_used="json-unless-none", return_type=u.Quantity),
]


def validate_path_type(v: str) -> Path:
    """Validation for Base Path Type Quantity"""
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


PathType = Annotated[
    str | Path,
    BeforeValidator(validate_path_type),
    PlainSerializer(lambda x: Path(x), when_used="json-unless-none", return_type=Path),
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


@dataclass
class TimeInterval:
    """
    Config section for getting main information for creating a Time Interval
    object.
    """

    interval: dict[str, str | float]

    def build(self) -> dict:
        value_dict = {}
        value_dict["format"] = Time(self.interval["start"]).format

        value_dict["start"] = str(self.interval["start"])

        value_dict["stop"] = str(self.interval["stop"])

        return value_dict

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type[dict], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        assert source is TimeInterval
        return core_schema.no_info_after_validator_function(
            cls._validate,
            core_schema.dict_schema(keys_schema=core_schema.str_schema(), values_schema=core_schema.str_schema()),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize,
                info_arg=False,
                return_schema=core_schema.dict_schema(
                    keys_schema=core_schema.str_schema(), values_schema=core_schema.str_schema()
                ),
            ),
        )

    @staticmethod
    def _validate(value: dict) -> "TimeInterval":
        inv_dict: dict[str, str | float] = {}

        if isinstance(value["format"], TimeFormatEnum):
            inv_dict["format"] = value["format"]

        # Read all values as string
        value["start"] = str(value["start"])
        value["stop"] = str(value["stop"])

        if not Time(value["start"], format=value["format"]):
            raise ValueError(f"{value['start']} is not the right Time value for format {value['format']}")
        else:
            inv_dict["start"] = Time(value["start"], format=value["format"])

        if not Time(value["stop"], format=value["format"]):
            raise ValueError(f"{value['stop']} is not the right Time value for format {value['format']}")
        else:
            inv_dict["stop"] = Time(value["stop"], format=value["format"])

        return TimeInterval(inv_dict)

    @staticmethod
    def _serialize(value: "TimeInterval") -> dict:
        return value.build()


class BaseConfig(BaseModel):
    """
    Base Config class for creating other Config sections with specific encoders.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        validate_default=True,
        use_enum_values=True,
    )

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"


# Basic Quantity ranges Type for building the Config
class EnergyRangeConfig(BaseConfig):
    """
    Config section for getting a energy range information for creating an
    Energy type Quantity object.
    """

    min: EnergyType = 1 * u.GeV
    max: EnergyType = 1 * u.TeV
