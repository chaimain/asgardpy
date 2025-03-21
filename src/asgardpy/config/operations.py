"""
Main AsgardpyConfig Operations Module
"""

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from gammapy.modeling.models import Models, SkyModel

__all__ = [
    "all_model_templates",
    "compound_model_dict_converstion",
    "get_model_template",
    "recursive_merge_dicts",
    "deep_update",
]

CONFIG_PATH = Path(__file__).resolve().parent

log = logging.getLogger(__name__)


def all_model_templates():
    """
    Collect all Template Models provided in Asgardpy, and their small tag names.
    """
    template_files = sorted(list(CONFIG_PATH.glob("model_templates/model_template*yaml")))

    all_tags = []
    for file in template_files:
        all_tags.append(file.name.split("_")[-1].split(".")[0])
    all_tags = np.array(all_tags)

    return all_tags, template_files


def get_model_template(spec_model_tag):
    """
    Read a particular template model yaml filename to create an AsgardpyConfig
    object.
    """
    all_tags, template_files = all_model_templates()
    new_model_file = None

    for file, tag in zip(template_files, all_tags, strict=True):
        if spec_model_tag == tag:
            new_model_file = file
    return new_model_file


def check_gammapy_model(gammapy_model):
    """
    For a given object type, try to read it as a Gammapy Models object.
    """
    if isinstance(gammapy_model, Models | SkyModel):
        models_gpy = Models(gammapy_model)
    else:
        try:
            models_gpy = Models.read(gammapy_model)
        except KeyError:
            raise TypeError("%s File cannot be read by Gammapy Models", gammapy_model) from KeyError

    return models_gpy


def recursive_merge_lists(final_config_key, extra_config_key, value):
    """
    Recursively merge from lists of dicts. Distinct function as an auxiliary for
    the recursive_merge_dicts function.
    """
    new_config = []

    for key_, value_ in zip(final_config_key, value, strict=False):
        key_ = recursive_merge_dicts(key_ or {}, value_)
        new_config.append(key_)

    # For example moving from a smaller list of model parameters to a
    # longer list.
    if len(final_config_key) < len(extra_config_key):
        for value_ in value[len(final_config_key) :]:
            new_config.append(value_)
    return new_config


def recursive_merge_dicts(base_config, extra_config):
    """
    Recursively merge two dictionaries.
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
            final_config[key] = recursive_merge_lists(final_config[key], extra_config[key], value)
        elif key in final_config and isinstance(final_config[key], dict):
            final_config[key] = recursive_merge_dicts(final_config.get(key) or {}, value)
        else:
            final_config[key] = value

    return final_config


def deep_update(d, u):
    """
    Recursively update a nested dictionary.

    Just like in Gammapy, taken from: https://stackoverflow.com/a/3233356/19802442
    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def compound_model_dict_converstion(dict):
    """
    Given a Gammapy CompoundSpectralModel as a dict object, convert it into
    an Asgardpy form.
    """
    ebl_abs = dict["model2"]
    ebl_abs["alpha_norm"] = ebl_abs["parameters"][0]["value"]
    ebl_abs["redshift"] = ebl_abs["parameters"][1]["value"]
    ebl_abs.pop("parameters", None)

    dict["type"] = dict["model1"]["type"]
    dict["parameters"] = dict["model1"]["parameters"]
    dict["ebl_abs"] = ebl_abs

    dict.pop("model1", None)
    dict.pop("model2", None)
    dict.pop("operator", None)

    return dict
