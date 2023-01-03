"""
Classes containing the DL4 products config parameters for the high-level interface
"""
from enum import Enum

from astropy import units as u
from gammapy.datasets import Datasets
from gammapy.estimators import FluxPointsEstimator, LightCurveEstimator
from gammapy.modeling import Fit
from gammapy.maps import Map

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
        self.log.info(self.fit_result)

    def _setup_fit(self):
        """
        Setup the Gammapy Fit function with all the provided parameters
        """
        self.fit = Fit(
            backend=self.fit_params.backend,
            optimize_opts = self.fit_params.optimize_opts,
            covariance_opts = self.fit_params.covariance_opts,
            confidence_opts = self.fit_params.confidence_opts,
            store_trace = self.fit_params.store_trace,
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

    """
    def plot_parameter_stat_profile(self, datasets, parameter, axs=None):
        total_stat = self.result.total_stat

        parameter.scan_n_values = 20

        profile = self.fit.stat_profile(datasets=datasets, parameter=parameter)

        axs.plot(profile[f"{parameter.name}_scan"], profile["stat_scan"] - total_stat)
        axs.set_xlabel(f"{parameter.unit}")
        axs.set_ylabel("Delta TS")
        axs.set_title(
            f"{parameter.name}: {parameter.value:.2e} +/- {parameter.error:.2e}"
            + f"\n{parameter.value:.2e}+/- {parameter.error/parameter.value:.2f}"
        )

        return axs

    def plot_spectrum_fp(self, axs=None, kwargs_fp=None):
        if kwargs_fp is None:
            kwargs_fp = {
                "sed_type": "e2dnde",
                "color": "black",
                "mfc": "gray",
                "marker": "D",
                "label": "Flux points",
            }
        self.flux_points.plot(ax=axs, **kwargs_fp)

        return axs

    def plot_lc(self, axs=None, kwargs=None):
        if kwargs is None:
            kwargs = {
                "sed_type": "flux",
                "axis_name": "time",
                "color": "black",
                "marker": "o",
                "label": "LC Flux points",
            }
        self.light_curve_flux.plot(ax=axs, **kwargs)

        return axs

    def plot_spectrum_model(self, axs=None, is_intrinsic=False, kwargs_model=None):
        energy_range = Quantity([self.energy_bin_edges[0], self.energy_bin_edges[-1]])

        if kwargs_model is None:
            kwargs_model = {
                "sed_type": "e2dnde",
                "color": "gray",
            }

        if is_intrinsic:
            spec = self.target_model.spectral_model.model1
            if "label" not in kwargs_model.keys():
                kwargs_model["label"] = "Best fit intrinsic model - EBL deabsorbed"
            spec.plot(ax=axs, energy_bounds=energy_range, **kwargs_model)
        else:
            spec = self.target_model.spectral_model
            spec.evaluate_error(energy_range)

            kwargs_model_err = kwargs_model.copy()
            kwargs_model_err.pop("label", None)

            spec.plot_error(ax=axs, energy_bounds=energy_range, **kwargs_model_err)
            spec.plot(ax=axs, energy_bounds=energy_range, **kwargs_model)

        return axs

    def plot_residuals(self, axs=None, method="diff", kwargs_res=None):
        self.flux_points_dataset = FluxPointsDataset(
            data=self.flux_points, models=SkyModel(spectral_model=self.target_model.spectral_model)
        )

        self.flux_points_dataset.plot_residuals(ax=axs, method=method, **kwargs_res)

        return axs

    def plot_ts_profiles(self, axs=None, add_cbar=True, kwargs_ts=None):
        if kwargs_ts is None:
            kwargs_ts = {
                "sed_type": "e2dnde",
                "color": "darkorange",
            }
        self.flux_points.plot_ts_profiles(ax=axs, add_cbar=add_cbar, **kwargs_ts)
        self.flux_points.plot(ax=axs, **kwargs_ts)

        return axs

    def plot_model_covariance_correlation(self, axs=None):
        spec = self.target_model.spectral_model
        spec.covariance.plot_correlation(ax=axs)

        return axs

    def plot_spectrum_enrico(
        self, axs=None, kwargs_fp=None, kwargs_model=None, kwargs_model_err=None
    ):
        y_mean = self.lat_bute["col2"]
        y_errs = self.lat_bute["col3"]

        y_errp = y_mean + y_errs
        y_errn = y_mean - y_errs  # 10**(2*np.log10(y_mean)-np.log10(y_errp))

        if kwargs_fp is None:
            kwargs_fp = {
                "marker": "o",
                "ls": "None",
                "color": "red",
                "mfc": "white",
                "lw": 0.75,
                "mew": 0.75,
                "zorder": -10,
            }

        if kwargs_model is None:
            kwargs_model = {
                "color": "red",
                "lw": 0.75,
                "mew": 0.75,
                "zorder": -10,
            }

        if kwargs_model_err is None:
            kwargs_model_err = kwargs_model.copy()
            kwargs_model_err.pop("mew", None)
            kwargs_model_err["alpha"] = 0.2
            kwargs_model_err["label"] = "Enrico/Fermitools"

        # Best-Fit model
        axs.plot(
            self.lat_bute["col1"] * u.MeV,
            y_mean * u.Unit("erg/(cm2*s)"),
            **kwargs_model,
        )
        # confidence band
        axs.fill_between(
            x=self.lat_bute["col1"] * u.MeV,
            y1=y_errn * u.Unit("erg/(cm2*s)"),
            y2=y_errp * u.Unit("erg/(cm2*s)"),
            **kwargs_model_err,
        )

        # spectral points
        # lat_ebin = self.lat_ebin # Copy?
        isuplim = self.lat_ebin["col5"] == 0
        self.lat_ebin["col5"][isuplim] = self.lat_ebin["col4"][isuplim] * 0.5
        kwargs_fp["uplims"] = isuplim

        axs.errorbar(
            x=self.lat_ebin["col1"] * u.MeV,
            y=self.lat_ebin["col4"] * u.Unit("erg/(cm2*s)"),
            xerr=[
                self.lat_ebin["col1"] * u.MeV - self.lat_ebin["col2"] * u.MeV,
                self.lat_ebin["col3"] * u.MeV - self.lat_ebin["col1"] * u.MeV,
            ],
            yerr=self.lat_ebin["col5"] * u.Unit("erg/(cm2*s)"),
            **kwargs_fp,
        )

        axs.set_ylim(
            [
                min(self.lat_ebin["col4"] * u.Unit("erg/(cm2*s)")) * 0.2,
                max(self.lat_ebin["col4"] * u.Unit("erg/(cm2*s)")) * 2,
            ]
        )

        return axs

    def plot_spectrum_fold(self, axs=None, fold_file=None, kwargs=None):
        if not os.path.exists(fold_file):
            return None

        if kwargs is None:
            kwargs = {
                "marker": "o",
                "ls": "None",
                "color": "C4",
                "mfc": "white",
                "zorder": -10,
                "label": "MAGIC/Fold",
                "lw": 0.75,
                "mew": 0.75,
                "alpha": 1,
            }

        fold = uproot.open(fold_file)
        fold_sed = fold["observed_sed"].tojson()

        f_x = fold_sed["fX"]
        f_y = fold_sed["fY"]
        x_err_low = fold_sed["fEXlow"]
        x_err_high = fold_sed["fEXhigh"]
        y_err_low = fold_sed["fEYlow"]
        y_err_high = fold_sed["fEYhigh"]

        axs.errorbar(
            x=f_x * u.GeV,
            y=f_y * u.Unit("TeV/(cm2 * s)"),
            xerr=[x_err_low * u.GeV, x_err_high * u.GeV],
            yerr=[y_err_low * u.Unit("TeV/(cm2 * s)"), y_err_high * u.Unit("TeV/(cm2 * s)")],
            **kwargs,
        )

        return axs
    """
