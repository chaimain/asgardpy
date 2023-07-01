import json
import logging
from enum import Enum
from pathlib import Path
from typing import List

import yaml
from gammapy.utils.scripts import make_path, read_yaml
from pydantic.utils import deep_update

from asgardpy.base import AnalysisStepEnum, BaseConfig, PathType
from asgardpy.data import (
    Dataset1DConfig,
    Dataset3DConfig,
    FitConfig,
    FluxPointsConfig,
    Target,
)

__all__ = [
    "AsgardpyConfig",
    "get_model_template",
    "recursive_merge_dicts",
]

CONFIG_PATH = Path(__file__).resolve().parent

log = logging.getLogger(__name__)


# Other general config params
class LogConfig(BaseConfig):
    level: str = "info"
    filename: str = ""
    filemode: str = "w"
    format: str = ""
    datefmt: str = ""


class ParallelBackendEnum(str, Enum):
    multi = "multiprocessing"
    ray = "ray"


class GeneralConfig(BaseConfig):
    log: LogConfig = LogConfig()
    outdir: PathType = PathType(".")
    n_jobs: int = 1
    parallel_backend: ParallelBackendEnum = ParallelBackendEnum.multi
    steps: List[AnalysisStepEnum] = []
    overwrite: bool = True
    stacked_dataset: bool = False


def get_model_template(spec_model_tag):
    """
    Read a particular template model yaml file into AsgardpyConfig object.
    """
    template_files = sorted(list(CONFIG_PATH.glob("model_template*yaml")))
    new_model_file = None
    for file in template_files:
        if spec_model_tag == file.name.split("_")[-1].split(".")[0]:
            new_model_file = file
    return new_model_file


def recursive_merge_dicts(a, b):
    """
    recursively merge two dictionaries.
    Entries in b override entries in a. The built-in update function cannot be
    used for hierarchical dicts.

    Also for the case when there is a list of dicts involved, one has to be more careful.

    Combined here are 2 options from SO.

    See:
    http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth/3233356#3233356
    and also
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth/18394648#18394648

    Parameters
    ----------
    a : dict
        dictionary to be merged
    b : dict
        dictionary to be merged
    Returns
    -------
    c : dict
        merged dict
    """
    c = a.copy()
    for k, v in b.items():
        if k in c and isinstance(c[k], list):
            new_c = []
            for cc, vv in zip(c[k], v):
                cc = recursive_merge_dicts(cc or {}, vv)
                new_c.append(cc)
            c[k] = new_c
        elif k in c and isinstance(c[k], dict):
            c[k] = recursive_merge_dicts(c.get(k) or {}, v)
        else:
            c[k] = v
    return c


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

    def update(self, config=None, merge_recursive=False):
        """
        Update config with provided settings.
        Parameters
        ----------
        config : string dict or `AsgardpyConfig` object
            The other configuration settings provided in dict() syntax.
        merge_recursive : bool
            Perform a recursive merge from the other config onto the parent config.

        Returns
        -------
        config : `AsgardpyConfig` object
            Updated config object.
        """
        if isinstance(config, str):
            other = AsgardpyConfig.from_yaml(config)
        elif isinstance(config, AsgardpyConfig):
            other = config
        else:
            raise TypeError(f"Invalid type: {config}")

        # Special case of when only updating target model parameters from a
        # separate file, where the name of the source is not provided.
        if other.target.components[0].name == "":
            merge_recursive = True

        if merge_recursive:
            config_new = recursive_merge_dicts(
                self.dict(exclude_defaults=True), other.dict(exclude_defaults=True)
            )
        else:
            config_new = deep_update(
                self.dict(exclude_defaults=True), other.dict(exclude_defaults=True)
            )
        return AsgardpyConfig(**config_new)
