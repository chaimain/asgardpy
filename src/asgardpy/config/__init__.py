"""
Configuration Module
"""

from asgardpy.config.generator import (
    AsgardpyConfig,
    GeneralConfig,
    gammapy_model_to_asgardpy_model_config,
    write_asgardpy_model_to_file,
)
from asgardpy.config.operations import (
    all_model_templates,
    compound_model_dict_converstion,
    deep_update,
    get_model_template,
    recursive_merge_dicts,
)

__all__ = [
    "all_model_templates",
    "compound_model_dict_converstion",
    "deep_update",
    "AsgardpyConfig",
    "GeneralConfig",
    "gammapy_model_to_asgardpy_model_config",
    "get_model_template",
    "recursive_merge_dicts",
    "write_asgardpy_model_to_file",
]
