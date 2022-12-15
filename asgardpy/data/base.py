"""classes containing the analysis steps supported by the high level interface"""

import abc
import logging
from enum import Enum

from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity
from pydantic import BaseModel

__all__ = [
    "AngleType",
    "EnergyType",
    "TimeType",
    "FrameEnum",
    "AnalysisStepBase",
    "AnalysisStep",
    "BaseConfig",
    "TimeRangeConfig",
    "EnergyRangeConfig",
]


class AngleType(Angle):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return Angle(v)


class EnergyType(Quantity):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        v = Quantity(v)
        if v.unit.physical_type != "energy":
            raise ValueError(f"Invalid unit for energy: {v.unit!r}")
        return v


class TimeType(Time):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return Time(v)


class FrameEnum(str, Enum):
    icrs = "icrs"
    galactic = "galactic"


class BaseConfig(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        json_encoders = {
            Angle: lambda v: f"{v.value} {v.unit}",
            Quantity: lambda v: f"{v.value} {v.unit}",
            Time: lambda v: f"{v.value}",
        }


class AnalysisStepBase(abc.ABC):
    tag = "analysis-step"

    def __init__(self, config, log=None, overwrite=True):  # name=None, 
        self.config = config
        self.overwrite = overwrite
        # self._name = make_name(name)

        if log is None:
            log = logging.getLogger(__name__)
            self.log = log

    # @property
    # def name(self):
    #    return self._name

    def run(self):  # , data
        # self.data = data
        # self.products = self.data
        self._run()

    @abc.abstractmethod
    def _run(self):
        pass


class AnalysisStep:
    "Create one of the analysis step class listed in the registry"

    @staticmethod
    def create(tag, config, **kwargs):
        from . import ANALYSIS_STEP_REGISTRY

        cls = ANALYSIS_STEP_REGISTRY.get_cls(tag)
        return cls(config, **kwargs)


class TimeRangeConfig(BaseConfig):
    start: TimeType = None
    stop: TimeType = None


class EnergyRangeConfig(BaseConfig):
    min: EnergyType = None
    max: EnergyType = None
