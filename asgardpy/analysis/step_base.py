"""
Base Classes for creating the intermediate High-level Analysis Steps
"""
import abc
import logging
from enum import Enum

__all__ = [
    "AnalysisStepBase",
    "AnalysisStepEnum",
]


class AnalysisStepBase(abc.ABC):
    """Config section for creating a basic AsgardpyAnalysis Step."""

    tag = "analysis-step"

    def __init__(self, config, log=None, overwrite=True):
        self.config = config
        self.overwrite = overwrite

        self.datasets = None
        self.instrument_spectral_info = None

        if log is None:
            log = logging.getLogger(__name__)
            self.log = log

    def run(self, datasets=None, instrument_spectral_info=None):
        """
        One can provide datasets and instrument_spectral_info to be used,
        especially for the High-level Analysis steps.
        """
        self.datasets = datasets
        self.instrument_spectral_info = instrument_spectral_info

        final_product = self._run()
        self.log.info("Analysis Step %s completed", self.tag)

        return final_product

    @abc.abstractmethod
    def _run(self):
        pass


class AnalysisStepEnum(str, Enum):
    """Config section for list of Analysis Steps."""

    datasets_1d = "datasets-1d"
    datasets_3d = "datasets-3d"
    fit = "fit"
    flux_points = "flux-points"
