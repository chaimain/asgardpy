"""
Main AsgardpyConfig Generator Module
"""

import json
import logging
import os
from enum import Enum
from pathlib import Path

import yaml
from gammapy.modeling.models import CompoundSpectralModel
from gammapy.utils.scripts import make_path, read_yaml

from asgardpy.analysis.step_base import AnalysisStepEnum
from asgardpy.base import BaseConfig, PathType
from asgardpy.config.operations import (
    CONFIG_PATH,
    check_gammapy_model,
    compound_model_dict_converstion,
    deep_update,
    recursive_merge_dicts,
)
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
    "gammapy_model_to_asgardpy_model_config",
    "write_asgardpy_model_to_file",
]

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
    outdir: PathType = "None"
    n_jobs: int = 1
    parallel_backend: ParallelBackendEnum = ParallelBackendEnum.multi
    steps: list[AnalysisStepEnum] = []
    overwrite: bool = True
    stacked_dataset: bool = False


def check_config(config):
    """
    For a given object type, try to read it as an AsgardpyConfig object.
    """
    if isinstance(config, str | Path):
        if Path(config).is_file():
            AConfig = AsgardpyConfig.read(config)
        else:
            AConfig = AsgardpyConfig.from_yaml(config)
    elif isinstance(config, AsgardpyConfig):
        AConfig = config
    else:
        raise TypeError(f"Invalid type: {config}")

    return AConfig


def gammapy_model_to_asgardpy_model_config(gammapy_model, asgardpy_config_file=None, recursive_merge=True):
    """
    Read the Gammapy Models object and save it as AsgardpyConfig object.

    The gammapy_model object may be a YAML config filename/path/object or a
    Gammapy Models object itself.

    Return
    ------
    asgardpy_config: `asgardpy.config.generator.AsgardpyConfig`
        Updated AsgardpyConfig object
    """

    models_gpy = check_gammapy_model(gammapy_model)

    models_gpy_dict = models_gpy.to_dict()

    if not asgardpy_config_file:
        asgardpy_config = AsgardpyConfig()  # Default object
        # Remove any name values in the model dict
        models_gpy_dict["components"][0].pop("datasets_names", None)
        models_gpy_dict["components"][0].pop("name", None)
    else:
        asgardpy_config = check_config(asgardpy_config_file)

    # For EBL part only
    if "model1" in models_gpy_dict["components"][0]["spectral"].keys():
        models_gpy_dict["components"][0]["spectral"] = compound_model_dict_converstion(
            models_gpy_dict["components"][0]["spectral"]
        )

    asgardpy_config_target_dict = asgardpy_config.model_dump()["target"]

    if recursive_merge:
        temp_target_dict = recursive_merge_dicts(asgardpy_config_target_dict, models_gpy_dict)
    else:
        # Use when there are nans present in the other config file, which are
        # the defaults in Gammapy, but NOT in Asgardpy.
        # E.g. test data Fermi-3fhl-crab model file
        temp_target_dict = deep_update(asgardpy_config_target_dict, models_gpy_dict)

    asgardpy_config.target = temp_target_dict

    return asgardpy_config


def write_asgardpy_model_to_file(gammapy_model, output_file=None, recursive_merge=True):
    """
    Read the Gammapy Models object and save it as AsgardpyConfig YAML file
    containing only the Model parameters, similar to the model templates
    available.
    """
    gammapy_model = check_gammapy_model(gammapy_model)

    asgardpy_config = gammapy_model_to_asgardpy_model_config(
        gammapy_model=gammapy_model[0],
        asgardpy_config_file=None,
        recursive_merge=recursive_merge,
    )

    if not output_file:
        if isinstance(gammapy_model[0].spectral_model, CompoundSpectralModel):
            model_tag = gammapy_model[0].spectral_model.model1.tag[1] + "_ebl"
        else:
            model_tag = gammapy_model[0].spectral_model.tag[1]

        output_file = CONFIG_PATH / f"model_templates/model_template_{model_tag}.yaml"
        os.path.expandvars(output_file)
    else:
        if not isinstance(output_file, Path):
            output_file = Path(os.path.expandvars(output_file))

    temp_ = asgardpy_config.model_dump(exclude_defaults=True)
    temp_["target"].pop("models_file", None)

    if isinstance(gammapy_model[0].spectral_model, CompoundSpectralModel):
        temp_["target"]["components"][0]["spectral"]["ebl_abs"]["filename"] = str(
            temp_["target"]["components"][0]["spectral"]["ebl_abs"]["filename"]
        )
    else:
        temp_["target"]["components"][0]["spectral"].pop("ebl_abs", None)

    yaml_ = yaml.dump(
        temp_,
        sort_keys=False,
        indent=4,
        width=80,
        default_flow_style=None,
    )

    output_file.write_text(yaml_)


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
            raise OSError(f"File exists already: {path}")
        path.write_text(self.to_yaml())

    def to_yaml(self):
        """
        Convert to YAML string.
        """
        # Here using `dict()` instead of `json()` would be more natural.
        # We should change this once pydantic adds support for custom encoders
        # to `dict()`. See https://github.com/samuelcolvin/pydantic/issues/1043
        data = json.loads(self.model_dump_json())
        return yaml.dump(data, sort_keys=False, indent=4, width=80, default_flow_style=None)

    def set_logging(self):
        """
        Set logging config.
        Calls ``logging.basicConfig``, i.e. adjusts global logging state.
        """
        self.general.log.level = self.general.log.level.upper()
        logging.basicConfig(**self.general.log.model_dump())
        log.info("Setting logging config: %s", self.general.log.model_dump())

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
        other = check_config(config)

        # Special case of when only updating target model parameters from a
        # separate file, where the name of the source is not provided.
        if other.target.components[0].name == "":
            merge_recursive = True

        if merge_recursive:
            config_new = recursive_merge_dicts(
                self.model_dump(exclude_defaults=True), other.model_dump(exclude_defaults=True)
            )
        else:
            config_new = deep_update(
                self.model_dump(exclude_defaults=True), other.model_dump(exclude_defaults=True)
            )
        return AsgardpyConfig(**config_new)
