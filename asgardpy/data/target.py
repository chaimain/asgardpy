"""
Classes containing the Target config parameters for the high-level interface
"""

from pathlib import Path

from asgardpy.data.base import BaseConfig
from asgardpy.data.geom import SkyCoordConfig

__all__ = [
    "EBLAbsorptionModel",
    "SpectralModelConfig",
    "SpatialModelConfig",
    "TargetModel",
    "TargetSource"
]


class EBLAbsorptionModel(BaseConfig):
    model_name: str = "dominguez"
    type: str = "EBLAbsorptionNormSpectralModel"
    alpha_norm: float = 1.0


class SpectralModelConfig(BaseConfig):
    model_name: str = "source_name"
    type: str = "type"
    parameters: dict = {}
    ebl_abs: EBLAbsorptionModel = EBLAbsorptionModel()


class SpatialModelConfig(BaseConfig):
    model_name: str = "model-name"
    type: str = "type"
    parameters: dict = {}


# Target information config
class TargetSource(BaseConfig):
    source_name: str = None
    sky_position: SkyCoordConfig = SkyCoordConfig()
    use_uniform_position: bool = True
    redshift: float = 0.0
    extended: bool = False


class TargetModel(BaseConfig):
    models_file: Path = None
    spectral: SpectralModelConfig = SpectralModelConfig()
    spatial: SpatialModelConfig = SpatialModelConfig()
