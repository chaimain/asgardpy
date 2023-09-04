"""
Module for inter-operable Gammapy objects
"""
from asgardpy.gammapy.interoperate_models import (
    get_gammapy_spectral_model,
    xml_spatial_model_to_gammapy,
    xml_spectral_model_to_gammapy,
)

__all__ = [
    "get_gammapy_spectral_model",
    "xml_spatial_model_to_gammapy",
    "xml_spectral_model_to_gammapy",
]
