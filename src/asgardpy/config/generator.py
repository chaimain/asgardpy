"""
Main AsgardpyConfig Generator Module
"""
import json
import logging
from enum import Enum
from pathlib import Path
from typing import List

import yaml
from gammapy.modeling.models import Models
from gammapy.utils.scripts import make_path, read_yaml
from pydantic.utils import deep_update

from asgardpy.analysis.step_base import AnalysisStepEnum
from asgardpy.base import BaseConfig, PathType
from asgardpy.data import (
    Dataset1DConfig,
    Dataset3DConfig,
    FitConfig,
    FluxPointsConfig,
    Target,
)

__all__ = [
    "AsgardpyConfig",
    "GeneralConfig",
    "gammapy_to_asgardpy_model_config",
    "get_model_template",
    "recursive_merge_dicts",
]

CONFIG_PATH = Path(__file__).resolve().parent

log = logging.getLogger(__name__)


# Other general config params
class LogConfig(BaseConfig):
    """Config section for main logging information."""

    level: str = "info"
    filename: str = ""
    filemode: str = "w"
    format: str = ""
    datefmt: str = ""


class ParallelBackendEnum(str, Enum):
    """Config section for list of parallel processing backend methods."""

    multi = "multiprocessing"
    ray = "ray"


class GeneralConfig(BaseConfig):
    """Config section for general information for running AsgardpyAnalysis."""

    log: LogConfig = LogConfig()
    outdir: PathType = PathType("None")
    n_jobs: int = 1
    parallel_backend: ParallelBackendEnum = ParallelBackendEnum.multi
    steps: List[AnalysisStepEnum] = []
    overwrite: bool = True
    stacked_dataset: bool = False


def get_model_template(spec_model_tag):
    """
    Read a particular template model yaml file into AsgardpyConfig object.
    """
    template_files = sorted(list(CONFIG_PATH.glob("model_templates/model_template*yaml")))
    new_model_file = None
    for file in template_files:
        if spec_model_tag == file.name.split("_")[-1].split(".")[0]:
            new_model_file = file
    return new_model_file


def recursive_merge_dicts(base_config, extra_config):
    """
    recursively merge two dictionaries.
    Entries in extra_config override entries in base_config. The built-in
    update function cannot be used for hierarchical dicts.

    Also for the case when there is a list of dicts involved, one has to be
    more careful. The extra_config may have longer list of dicts as compared
    with the base_config, in which case, the extra items are simply added to
    the merged final list.

    Combined here are 2 options from SO.

    See:
    http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth/3233356#3233356
    and also
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth/18394648#18394648

    Parameters
    ----------
    base_config : dict
        dictionary to be merged
    extra_config : dict
        dictionary to be merged
    Returns
    -------
    final_config : dict
        merged dict
    """
    final_config = base_config.copy()

    for key, value in extra_config.items():
        if key in final_config and isinstance(final_config[key], list):
            new_config = []

            for key_, value_ in zip(final_config[key], value):
                key_ = recursive_merge_dicts(key_ or {}, value_)
                new_config.append(key_)

            # For example moving from a smaller list of model parameters to a
            # longer list.
            if len(final_config[key]) < len(extra_config[key]):
                for value_ in value[len(final_config[key]) :]:
                    new_config.append(value_)
            final_config[key] = new_config

        elif key in final_config and isinstance(final_config[key], dict):
            final_config[key] = recursive_merge_dicts(final_config.get(key) or {}, value)
        else:
            final_config[key] = value

    return final_config


def gammapy_to_asgardpy_model_config(gammapy_model, asgardpy_config_file=None, recursive_merge=True):
    """
    Read the Gammapy Models YAML file and save it as AsgardpyConfig object.

    Return
    ------
    asgardpy_config: `asgardpy.config.generator.AsgardpyConfig`
        Updated AsgardpyConfig object
    """
    try:
        models_gpy = Models.read(gammapy_model)
    except KeyError:
        log.error("%s File cannot be read by Gammapy Models", gammapy_model)
        return None

    if not asgardpy_config_file:
        asgardpy_config = AsgardpyConfig()  # Default object
    elif isinstance(asgardpy_config_file, str):  # File path
        asgardpy_config = AsgardpyConfig.read(asgardpy_config_file)
    elif isinstance(asgardpy_config_file, AsgardpyConfig):
        asgardpy_config = asgardpy_config_file
    # also for YAML object?

    models_gpy_dict = models_gpy.to_dict()
    asgardpy_config_target_dict = asgardpy_config.dict()["target"]

    if recursive_merge:
        temp_target_dict = recursive_merge_dicts(asgardpy_config_target_dict, models_gpy_dict)
    else:
        # Use when there are nans present in the other config file, which are
        # the defaults in Gammapy, but NOT in Asgardpy.
        # E.g. test data Fermi-3fhl-crab model file
        temp_target_dict = deep_update(asgardpy_config_target_dict, models_gpy_dict)
    asgardpy_config.target = temp_target_dict

    return asgardpy_config


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
            config_new = recursive_merge_dicts(self.dict(exclude_defaults=True), other.dict(exclude_defaults=True))
        else:
            config_new = deep_update(self.dict(exclude_defaults=True), other.dict(exclude_defaults=True))
        return AsgardpyConfig(**config_new)
