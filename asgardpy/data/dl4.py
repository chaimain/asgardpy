"""
Main classes to define High-level Analysis Config and the Analysis Steps.
"""

from enum import Enum

from astropy import units as u
from astropy.time import Time
from gammapy.datasets import Datasets
from gammapy.estimators import FluxPointsEstimator, LightCurveEstimator
from gammapy.modeling import Fit

from asgardpy.data.base import (
    AnalysisStepBase,
    AngleType,
    BaseConfig,
    EnergyRangeConfig,
    TimeIntervalsConfig,
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


# Defining various components of High-level Analysis Config
class BackendEnum(str, Enum):
    minuit = "minuit"
    scipy = "scipy"


class FitConfig(BaseConfig):
    fit_range: EnergyRangeConfig = EnergyRangeConfig()
    backend: BackendEnum = BackendEnum.minuit
    optimize_opts: dict = {}
    covariance_opts: dict = {}
    confidence_opts: dict = {}
    store_trace: bool = True


class FluxPointsConfig(BaseConfig):
    parameters: dict = {"selection_optional": "all"}


class LightCurveConfig(BaseConfig):
    time_intervals: TimeIntervalsConfig = TimeIntervalsConfig()
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()
    parameters: dict = {"selection_optional": "all"}


class ExcessMapConfig(BaseConfig):
    correlation_radius: AngleType = 0.1 * u.deg
    parameters: dict = {}
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()


# The main Analysis Steps
class FitAnalysisStep(AnalysisStepBase):
    """
    Using the Fitting parameters as defined in the Config, with the given
    datasets perform the fit of the models to the updated list of datasets.
    """

    tag = "fit"

    def _run(self):
        self.fit_params = self.config.fit_params

        self._setup_fit()
        final_dataset = self._set_datasets()
        self.fit_result = self.fit.run(datasets=final_dataset)

        self.log.info(self.fit_result)

    def _setup_fit(self):
        """
        Setup the Gammapy Fit function with all the provided parameters from
        the config.
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
    Using the Flux Points Estimator parameters in the config, and the given
    datasets and instrument_spectral_info perform the Flux Points Estimation
    and store the result in a list of flux points for each dataset.
    """

    tag = "flux-points"

    def _run(self):
        self.flux_points = []
        datasets, energy_edges = self._sort_datasets_info()

        for dataset, energy_edges in zip(datasets, energy_edges):
            self._set_fpe(dataset, energy_edges)
            flux_points = self.fpe.run(datasets=dataset)
            flux_points.name = dataset.names

            self.flux_points.append(flux_points)

    def _set_fpe(self, dataset, energy_bin_edges):
        """
        Setup the Gammapy FluxPointsEstimator function with all the
        provided parameters.
        """
        fpe_settings = self.config.flux_points_params.parameters

        self.fpe = FluxPointsEstimator(
            energy_edges=energy_bin_edges, source=self.config.target.source_name, **fpe_settings
        )

    def _sort_datasets_info(self):
        """
        The given list of datasets may contain sub-instrument level datasets.
        With the help of the dict information for instrument specific name and
        spectral energy edges, this function, sorts the datasets and returns
        them to be passed to the Flux Points Estimator function.

        Returns
        -------
        sorted_datasets: List of Datasets object.
        sorted_energy_edges: List of energy edges for flux points estimation
            for respective instruments' datasets
        """
        dataset_name_list = self.datasets.names
        sorted_datasets = []
        sorted_energy_edges = []

        for i, name in enumerate(self.instrument_spectral_info["name"]):
            dataset_list = []
            for j, dataset_names in enumerate(dataset_name_list):
                if name in dataset_names:
                    dataset_list.append(self.datasets[j])
            if len(dataset_list) != 0:
                sorted_energy_edges.append(
                    self.instrument_spectral_info["spectral_energy_ranges"][i]
                )
                dataset_list = Datasets(dataset_list)
                sorted_datasets.append(dataset_list)

        return sorted_datasets, sorted_energy_edges


class LightCurveAnalysisStep(AnalysisStepBase):
    """
    Using the Light Curve Estimator parameters in the config, and the given
    datasets and instrument_spectral_info perform the Light Curve Estimation
    and store the result in a list of flux points for each dataset.
    """

    tag = "light-curve"

    def _run(self):
        self.light_curve = []

        for dataset in self.datasets:
            self._set_lce(dataset=dataset)
            light_curve = self.lce.run(datasets=dataset)
            light_curve.name = dataset.name

            self.light_curve.append(light_curve)

    def _set_lce(self, dataset=None):
        """
        Setup the Gammapy LightCurveEstimator function with all the
        provided parameters.
        """
        if dataset is None:
            dataset = self.datasets

        energy_range = self.config.light_curve_params.energy_edges
        energy_min = u.Quantity(energy_range.min)
        energy_max = u.Quantity(energy_range.max)

        # Check with the given energy range of counts of each dataset.
        dataset_energy = dataset.counts.geom.axes["energy"].edges
        data_geom_energy_min = dataset_energy[0]
        data_geom_energy_max = dataset_energy[-1]

        # Fix the energy range to be within the given dataset.
        if energy_min < data_geom_energy_min:
            energy_min = data_geom_energy_min
        if energy_max > data_geom_energy_max:
            energy_max = data_geom_energy_max

        energy_bin_edges = [energy_min, energy_max]

        time_intervals_params = self.config.light_curve_params.time_intervals
        if time_intervals_params.intervals[0].start is None:
            self.log.info("Time intervals not defined. Extract light curve on datasets GTIs.")
            time_intervals = None
        else:
            time_intervals = []
            for interval in time_intervals_params.intervals:
                time_intervals.append(
                    [
                        Time(interval.start, format=time_intervals_params.format),
                        Time(interval.stop, format=time_intervals_params.format),
                    ]
                )
        lce_settings = self.config.light_curve_params.parameters

        self.lce = LightCurveEstimator(
            energy_edges=energy_bin_edges,
            time_intervals=time_intervals,
            source=self.config.target.source_name,
            **lce_settings
        )
