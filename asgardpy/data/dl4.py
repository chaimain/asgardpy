"""
Main classes to define High-level Analysis Config and the Analysis Steps.
"""

from enum import Enum

from astropy import units as u
from gammapy.datasets import Datasets
from gammapy.estimators import FluxPointsEstimator
from gammapy.modeling import Fit

from asgardpy.base import AnalysisStepBase, BaseConfig, EnergyRangeConfig

__all__ = [
    "FitAnalysisStep",
    "FitConfig",
    "FluxPointsAnalysisStep",
    "FluxPointsConfig",
]


# Defining various components of High-level Analysis Config
class BackendEnum(str, Enum):
    """Config section for a list Fitting backend methods."""

    minuit = "minuit"
    scipy = "scipy"


class FitConfig(BaseConfig):
    """Config section for parameters to use for Fit function."""

    fit_range: EnergyRangeConfig = EnergyRangeConfig()
    backend: BackendEnum = BackendEnum.minuit
    optimize_opts: dict = {}
    covariance_opts: dict = {}
    confidence_opts: dict = {}
    store_trace: bool = True


class FluxPointsConfig(BaseConfig):
    """Config section for parameters to use for FluxPointsEstimator function."""

    parameters: dict = {"selection_optional": "all"}
    reoptimize: bool = False


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

        self.instrument_spectral_info["total_stat"] = self.fit_result.total_stat

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
            self._set_fpe(energy_edges)
            flux_points = self.fpe.run(datasets=dataset)
            flux_points.name = dataset.names

            self.flux_points.append(flux_points)

    def _set_fpe(self, energy_bin_edges):
        """
        Setup the Gammapy FluxPointsEstimator function with all the
        provided parameters.
        """
        fpe_settings = self.config.flux_points_params.parameters

        self.fpe = FluxPointsEstimator(
            energy_edges=energy_bin_edges,
            source=self.config.target.source_name,
            n_jobs=self.config.general.n_jobs,
            parallel_backend=self.config.general.parallel_backend,
            reoptimize=self.config.flux_points_params.reoptimize,
            **fpe_settings,
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
