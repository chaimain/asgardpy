import json
import logging
from pathlib import Path
from typing import List

import yaml
from gammapy.utils.scripts import make_path, read_yaml
from pydantic.utils import deep_update

from asgardpy.data.base import AnalysisStepEnum, BaseConfig
from asgardpy.data.dataset_1d import Dataset1DConfig
from asgardpy.data.dataset_3d import Dataset3DConfig
from asgardpy.data.dl4 import (
    ExcessMapConfig,
    FitConfig,
    FluxPointsConfig,
    LightCurveConfig,
)
from asgardpy.data.target import Target

__all__ = ["AsgardpyConfig"]

CONFIG_PATH = Path(__file__).resolve().parent / "config"
DOCS_FILE = CONFIG_PATH / "docs.yaml"

log = logging.getLogger(__name__)


# Other general config params
class LogConfig(BaseConfig):
    level: str = "info"
    filename: Path = Path(".")
    filemode: str = ""
    format: str = ""
    datefmt: str = ""


class GeneralConfig(BaseConfig):
    log: LogConfig = LogConfig()
    outdir: str = "."
    n_jobs: int = 1
    steps: List[AnalysisStepEnum] = []
    overwrite: bool = True
    stacked_dataset: bool = False


# Combine everything!
class AsgardpyConfig(BaseConfig):
    """
    Asgardpy analysis configuration, based on Gammapy Analysis Config.
    """

    general: GeneralConfig = GeneralConfig()

    target: Target = Target()

    dataset3d: Dataset3DConfig = Dataset3DConfig()
    dataset1d: Dataset1DConfig = Dataset1DConfig()

    fit_params: FitConfig = FitConfig()
    flux_points_params: FluxPointsConfig = FluxPointsConfig()
    excess_map_params: ExcessMapConfig = ExcessMapConfig()
    light_curve_params: LightCurveConfig = LightCurveConfig()

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
