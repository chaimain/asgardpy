"""
Module for inter-operable Gammapy objects
"""
from asgardpy.gammapy.interoperate_models import (
    xml_spatial_model_to_gammapy,
    xml_spectral_model_to_gammapy,
)

__all__ = [
    "xml_spatial_model_to_gammapy",
    "xml_spectral_model_to_gammapy",
]
