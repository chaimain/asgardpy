"""
Classes containing the Target config parameters for the high-level interface
"""

from pathlib import Path
from typing import List

from asgardpy.data.base import BaseConfig
from asgardpy.data.geom import SkyCoordConfig

__all__ = [
    "EBLAbsorptionModel",
    "SpectralModelConfig",
    "SpatialModelConfig",
    "Target"
]


class EBLAbsorptionModel(BaseConfig):
    model_name: str = "dominguez"
    type: str = "EBLAbsorptionNormSpectralModel"
    redshift: float = 0.4
    alpha_norm: float = 1.0


class ModelParams(BaseConfig):
    name: str = None
    value: float = None
    unit: str = None
    error: float = None
    min: float = None
    max: float = None
    frozen: bool = True


class SpectralModelConfig(BaseConfig):
    model_name: str = None
    type: str = None
    parameters: List[ModelParams] = [ModelParams()]
    ebl_abs: EBLAbsorptionModel = EBLAbsorptionModel()


class SpatialModelConfig(BaseConfig):
    model_name: str = None
    type: str = None
    parameters: List[ModelParams] = [ModelParams()]


class SkyModelComponent(BaseConfig):
    name: str = None
    type: str = "SkyModel"
    spectral: SpectralModelConfig = SpectralModelConfig()
    spatial: SpatialModelConfig = SpatialModelConfig()


class Target(BaseConfig):
    source_name: str = None
    sky_position: SkyCoordConfig = SkyCoordConfig()
    use_uniform_position: bool = True
    models_file: Path = None
    extended: bool = False
    components: SkyModelComponent = SkyModelComponent()
    covariance: str = None
    from_fermi: bool = False
