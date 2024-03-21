"""
Main Class for the intermediate High-level Analysis Steps
"""

from gammapy.utils.registry import Registry

from asgardpy.data import (
    Datasets1DAnalysisStep,
    Datasets3DAnalysisStep,
    FitAnalysisStep,
    FluxPointsAnalysisStep,
)

__all__ = ["AnalysisStep"]

ANALYSIS_STEP_REGISTRY = Registry(
    [
        Datasets1DAnalysisStep,
        Datasets3DAnalysisStep,
        FitAnalysisStep,
        FluxPointsAnalysisStep,
    ]
)


class AnalysisStep:
    """
    Base class for creating Asgardpy Analysis Steps.
    """

    @staticmethod
    def create(tag, config, **kwargs):
        """
        Create one of the Analysis Step class listed in the Registry.
        """
        cls = ANALYSIS_STEP_REGISTRY.get_cls(tag)
        return cls(config, **kwargs)
