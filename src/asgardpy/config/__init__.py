"""
Configuration Module
"""

from asgardpy.config.generator import (
    AsgardpyConfig,
    GeneralConfig,
    all_model_templates,
    gammapy_to_asgardpy_model_config,
    get_model_template,
    recursive_merge_dicts,
)

__all__ = [
    "all_model_templates",
    "AsgardpyConfig",
    "GeneralConfig",
    "gammapy_to_asgardpy_model_config",
    "get_model_template",
    "recursive_merge_dicts",
]
