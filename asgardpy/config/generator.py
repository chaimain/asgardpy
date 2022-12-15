import json
import logging

# from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List
from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity
import yaml
from pydantic import BaseModel
from pydantic.utils import deep_update
from gammapy.utils.scripts import make_path, read_yaml
from asgardpy.data import Dataset1DConfig, Dataset3DConfig

__all__ = ["BaseConfig", "AsgardpyConfig"]

CONFIG_PATH = Path(__file__).resolve().parent / "config"
DOCS_FILE = CONFIG_PATH / "docs.yaml"

log = logging.getLogger(__name__)


# Basic input classes
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


# Start to enlist info on the different datasets
# Main Base config schema
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


# Auxillary config info
class SkyCoordConfig(BaseConfig):
    frame: FrameEnum = None
    lon: AngleType = None
    lat: AngleType = None


class EnergyAxisConfig(BaseConfig):
    min: EnergyType = None
    max: EnergyType = None
    nbins: int = None


class EnergyRangeConfig(BaseConfig):
    min: EnergyType = None
    max: EnergyType = None


class TimeRangeConfig(BaseConfig):
    start: TimeType = None
    stop: TimeType = None


# DL4 products config
class FluxPointsConfig(BaseConfig):
    energy: EnergyAxisConfig = EnergyAxisConfig()
    source: str = "source"
    parameters: dict = {"selection_optional": "all"}


class LightCurveConfig(BaseConfig):
    time_intervals: TimeRangeConfig = TimeRangeConfig()
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()
    source: str = "source"
    parameters: dict = {"selection_optional": "all"}


class FitConfig(BaseConfig):
    fit_range: EnergyRangeConfig = EnergyRangeConfig()


class ExcessMapConfig(BaseConfig):
    correlation_radius: AngleType = "0.1 deg"
    parameters: dict = {}
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()


# Target Model info config
class EBLAbsorptionModel(BaseConfig):
    model_name: str = "dominguez"
    type: str = "EBLAbsorptionNormSpectralModel"
    alpha_norm: float = 1.0


class SpectralModelConfig(BaseConfig):
    model_name: str = "source_name"
    type: str = "type"
    parameters: dict = {}
    ebl_abs: EBLAbsorptionModel = EBLAbsorptionModel()


class SpatialModelConfig(BaseConfig):
    model_name: str = "model-name"
    type: str = "type"
    parameters: dict = {}


# Target information config
class TargetSource(BaseConfig):
    source_name: str = None
    sky_position: SkyCoordConfig = SkyCoordConfig()
    use_uniform_position: bool = True
    redshift: float = 0.0
    extended: bool = False


class TargetModel(BaseConfig):
    models_file: Path = None
    spectral: SpectralModelConfig = SpectralModelConfig()
    spatial: SpatialModelConfig = SpatialModelConfig()


# Other general config params
class LogConfig(BaseConfig):
    level: str = "info"
    filename: List[Path] = []
    filemode: List[str] = []
    format: str = None
    datefmt: str = None


class GeneralConfig(BaseConfig):
    log: LogConfig = LogConfig()
    outdir: str = "."
    n_jobs: int = 1


# Combine everything!
class AsgardpyConfig(BaseConfig):
    """
    Asgardpy analysis configuration, based on Gammapy Analysis Config.
    """

    general: GeneralConfig = GeneralConfig()

    source: TargetSource = TargetSource()
    model: TargetModel = TargetModel()

    dataset3d: Dataset3DConfig = Dataset3DConfig()
    dataset1d: Dataset1DConfig = Dataset1DConfig()

    fit: FitConfig = FitConfig()
    flux_points: FluxPointsConfig = FluxPointsConfig()
    excess_map: ExcessMapConfig = ExcessMapConfig()
    light_curve: LightCurveConfig = LightCurveConfig()

    def __str__(self):
        """
        Display settings in pretty YAML format.
        """
        info = self.__class__.__name__ + "\n\n\t"
        data = self.to_yaml()
        data = data.replace("\n", "\n\t")
        info += data
        return info.expandtabs(tabsize=4)

    @classmethod
    def read(cls, path):
        """
        Reads from YAML file.
        """
        config = read_yaml(path)
        return AsgardpyConfig(**config)

    @classmethod
    def from_yaml(cls, config_str):
        """
        Create from YAML string.
        """
        settings = yaml.safe_load(config_str)
        return AsgardpyConfig(**settings)

    def write(self, path, overwrite=False):
        """
        Write to YAML file.
        """
        path = make_path(path)
        if path.exists() and not overwrite:
            raise IOError(f"File exists already: {path}")
        path.write_text(self.to_yaml())

    def to_yaml(self):
        """
        Convert to YAML string.
        """
        # Here using `dict()` instead of `json()` would be more natural.
        # We should change this once pydantic adds support for custom encoders
        # to `dict()`. See https://github.com/samuelcolvin/pydantic/issues/1043
        config = json.loads(self.json())
        return yaml.dump(config, sort_keys=False, indent=4, width=80, default_flow_style=None)

    def set_logging(self):
        """
        Set logging config.
        Calls ``logging.basicConfig``, i.e. adjusts global logging state.
        """
        self.general.log.level = self.general.log.level.upper()
        logging.basicConfig(**self.general.log.dict())
        log.info("Setting logging config: {!r}".format(self.general.log.dict()))

    def update(self, config=None):
        """
        Update config with provided settings.
        Parameters
        ----------
        config : string dict or `AsgardpyConfig` object
            Configuration settings provided in dict() syntax.
        """
        if isinstance(config, str):
            other = AsgardpyConfig.from_yaml(config)
        elif isinstance(config, AsgardpyConfig):
            other = config
        else:
            raise TypeError(f"Invalid type: {config}")

        config_new = deep_update(
            self.dict(exclude_defaults=True), other.dict(exclude_defaults=True)
        )
        return AsgardpyConfig(**config_new)
