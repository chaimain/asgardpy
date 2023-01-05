"""
Classes containing the DL4 products config parameters for the high-level interface
"""
from enum import Enum

from astropy import units as u
from gammapy.datasets import Datasets
from gammapy.estimators import FluxPointsEstimator, LightCurveEstimator

# from gammapy.maps import Map
from gammapy.modeling import Fit

from asgardpy.data.base import (
    AnalysisStepBase,
    AngleType,
    BaseConfig,
    EnergyRangeConfig,
    TimeRangeConfig,
)
from asgardpy.data.geom import EnergyAxisConfig

__all__ = [
    "FluxPointsConfig",
    "LightCurveConfig",
    "FitConfig",
    "ExcessMapConfig",
    "FitAnalysisStep",
    "FluxPointsAnalysisStep",
    "LightCurveAnalysisStep",
]


class FluxPointsConfig(BaseConfig):
    energy: EnergyAxisConfig = EnergyAxisConfig()
    parameters: dict = {"selection_optional": "all"}


class LightCurveConfig(BaseConfig):
    time_intervals: TimeRangeConfig = TimeRangeConfig()
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()
    parameters: dict = {"selection_optional": "all"}


class BackendEnum(str, Enum):
    minuit = "minuit"
    scipy = "scipy"


class FitConfig(BaseConfig):
    fit_range: EnergyRangeConfig = EnergyRangeConfig()
    backend: BackendEnum = None
    optimize_opts: dict = {}
    covariance_opts: dict = {}
    confidence_opts: dict = {}
    store_trace: bool = True


class ExcessMapConfig(BaseConfig):
    correlation_radius: AngleType = "0.1 deg"
    parameters: dict = {}
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()


class FitAnalysisStep(AnalysisStepBase):
    """
    Fit the target model to the updated list of datasets.
    """

    tag = "fit"

    def _run(self):
        self.fit_params = self.config.fit_params

        self._setup_fit()
        final_dataset = self._set_datasets()
        self.fit_result = self.fit.run(datasets=final_dataset)
        best_fit_model = final_dataset.models.to_dict()
        self.log.info(self.fit_result)
        self.log.info(best_fit_model)

    def _setup_fit(self):
        """
        Setup the Gammapy Fit function with all the provided parameters
        """
        self.fit = Fit(
            backend=self.fit_params.backend,
            optimize_opts=self.fit_params.optimize_opts,
            covariance_opts=self.fit_params.covariance_opts,
            confidence_opts=self.fit_params.confidence_opts,
            store_trace=self.fit_params.store_trace,
        )

    def _set_datasets(self):
        """
        Prepare each dataset for running the Fit function, by setting the
        energy range.
        """
        en_min = u.Quantity(self.fit_params.fit_range.min)
        en_max = u.Quantity(self.fit_params.fit_range.max)

        final_dataset = Datasets()
        for data in self.datasets:
            geom = data.counts.geom
            data.mask_fit = geom.energy_mask(en_min, en_max)
            final_dataset.append(data)

        return final_dataset


class FluxPointsAnalysisStep(AnalysisStepBase):
    """
    Retrieve flux points for a given dataset.
    Currently getting flux points for ALL datasets and storing them in a list.
    """

    tag = "flux-points"

    def _run(self):
        self.flux_points = []

        for dataset in self.datasets:
            self.flux_points.append(self._get_spectral_points(dataset))

    def get_spectral_points(self, datasets):
        """ """
        energy_bin_edges = self.config.flux_points_params.energy

        fpe_settings = self.config.flux_points_params
        fpe_settings.pop("energy")

        fpe = FluxPointsEstimator(
            energy_edges=energy_bin_edges, source=self.config.target.source_name, **fpe_settings
        )

        flux_points = fpe.run(datasets=datasets)

        return flux_points


class LightCurveAnalysisStep(AnalysisStepBase):
    """
    Retrieve light curve flux points for a given dataset.
    Currently getting flux points for ALL datasets and storing them in a list.
    """

    tag = "light-curve"

    def _run(self):
        self.light_curve = []

        for dataset in self.datasets:
            self.light_curve.append(self._get_lc_flux_points(dataset))

    def get_lc_flux_points(self, datasets=None):
        """ """
        energy_range = self.config.light_curve_params.energy_range
        time_intervals = self.config.light_curve_params.time_intervals

        lc_settings = self.config.light_curve_params
        lc_settings.pop("energy_range")
        lc_settings.pop("time_intervals")

        lc_flux = LightCurveEstimator(
            energy_edges=energy_range,
            time_intervals=time_intervals,
            source=self.config.target.source_name,
            **lc_settings
        )

        light_curve_flux = lc_flux.run(datasets=datasets)

        return light_curve_flux
