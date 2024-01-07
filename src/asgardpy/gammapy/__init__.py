"""
Module for inter-operable Gammapy objects
"""
from asgardpy.gammapy.interoperate_models import (
    get_gammapy_spectral_model,
    xml_spatial_model_to_gammapy,
    xml_spectral_model_to_gammapy,
)
from asgardpy.gammapy.read_models import (
    create_gal_diffuse_skymodel,
    create_iso_diffuse_skymodel,
    create_source_skymodel,
    read_fermi_xml_models_list,
    update_aux_info_from_fermi_xml,
)

__all__ = [
    "get_gammapy_spectral_model",
    "xml_spatial_model_to_gammapy",
    "xml_spectral_model_to_gammapy",
    "create_gal_diffuse_skymodel",
    "create_iso_diffuse_skymodel",
    "create_source_skymodel",
    "read_fermi_xml_models_list",
    "update_aux_info_from_fermi_xml",
]
