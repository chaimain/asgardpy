"""
Some plotting functions to aid the high-level analysis.
"""
import os

# import matplotlib
import uproot
from astropy import units as u
from gammapy.datasets import FluxPointsDataset
from gammapy.modeling.models import SkyModel

__all__ = [
    "plot_parameter_stat_profile",
    "plot_spectrum_fp",
    "plot_lc",
    "plot_spectrum_model",
    "plot_residuals",
    "plot_ts_profiles",
    "plot_model_covariance_correlation",
    "plot_spectrum_enrico",
    "plot_spectrum_fold",
]


def plot_parameter_stat_profile(datasets, fit, total_stat, parameter, axs=None):
    # total_stat = self.result.total_stat

    parameter.scan_n_values = 20

    profile = fit.stat_profile(datasets=datasets, parameter=parameter)

    axs.plot(profile[f"{parameter.name}_scan"], profile["stat_scan"] - total_stat)
    axs.set_xlabel(f"{parameter.unit}")
    axs.set_ylabel("Delta TS")
    axs.set_title(
        f"{parameter.name}: {parameter.value:.2e} +/- {parameter.error:.2e}"
        + f"\n{parameter.value:.2e}+/- {parameter.error/parameter.value:.2f}"
    )

    return axs


def plot_spectrum_fp(flux_points, axs=None, kwargs_fp=None):
    if kwargs_fp is None:
        kwargs_fp = {
            "sed_type": "e2dnde",
            "color": "black",
            "mfc": "gray",
            "marker": "D",
            "label": "Flux points",
        }
    flux_points.plot(ax=axs, **kwargs_fp)

    return axs


def plot_lc(light_curve_flux, axs=None, kwargs=None):
    if kwargs is None:
        kwargs = {
            "sed_type": "flux",
            "axis_name": "time",
            "color": "black",
            "marker": "o",
            "label": "LC Flux points",
        }
    light_curve_flux.plot(ax=axs, **kwargs)

    return axs


def plot_spectrum_model(
    spectral_model, energy_range, axs=None, is_intrinsic=False, kwargs_model=None
):
    # energy_range = Quantity([self.energy_bin_edges[0], self.energy_bin_edges[-1]])

    if kwargs_model is None:
        kwargs_model = {
            "sed_type": "e2dnde",
            "color": "gray",
        }

    if is_intrinsic:
        spec = spectral_model.model1
        if "label" not in kwargs_model.keys():
            kwargs_model["label"] = "Best fit intrinsic model - EBL deabsorbed"
        spec.plot(ax=axs, energy_bounds=energy_range, **kwargs_model)
    else:
        spec = spectral_model
        spec.evaluate_error(energy_range)

        kwargs_model_err = kwargs_model.copy()
        kwargs_model_err.pop("label", None)

        spec.plot_error(ax=axs, energy_bounds=energy_range, **kwargs_model_err)
        spec.plot(ax=axs, energy_bounds=energy_range, **kwargs_model)

    return axs


def plot_residuals(flux_points, spectral_model, axs=None, method="diff", kwargs_res=None):
    flux_points_dataset = FluxPointsDataset(
        data=flux_points, models=SkyModel(spectral_model=spectral_model)
    )

    flux_points_dataset.plot_residuals(ax=axs, method=method, **kwargs_res)

    return axs


def plot_ts_profiles(flux_points, axs=None, add_cbar=True, kwargs_ts=None):
    if kwargs_ts is None:
        kwargs_ts = {
            "sed_type": "e2dnde",
            "color": "darkorange",
        }
    flux_points.plot_ts_profiles(ax=axs, add_cbar=add_cbar, **kwargs_ts)
    flux_points.plot(ax=axs, **kwargs_ts)

    return axs


def plot_model_covariance_correlation(target_model, axs=None):
    spec = target_model.spectral_model
    spec.covariance.plot_correlation(ax=axs)

    return axs


def plot_spectrum_enrico(
    lat_bute, lat_ebin, axs=None, kwargs_fp=None, kwargs_model=None, kwargs_model_err=None
):
    y_mean = lat_bute["col2"]
    y_errs = lat_bute["col3"]

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
        lat_bute["col1"] * u.MeV,
        y_mean * u.Unit("erg/(cm2*s)"),
        **kwargs_model,
    )
    # confidence band
    axs.fill_between(
        x=lat_bute["col1"] * u.MeV,
        y1=y_errn * u.Unit("erg/(cm2*s)"),
        y2=y_errp * u.Unit("erg/(cm2*s)"),
        **kwargs_model_err,
    )

    # spectral points
    # lat_ebin = self.lat_ebin # Copy?
    isuplim = lat_ebin["col5"] == 0
    lat_ebin["col5"][isuplim] = lat_ebin["col4"][isuplim] * 0.5
    kwargs_fp["uplims"] = isuplim

    axs.errorbar(
        x=lat_ebin["col1"] * u.MeV,
        y=lat_ebin["col4"] * u.Unit("erg/(cm2*s)"),
        xerr=[
            lat_ebin["col1"] * u.MeV - lat_ebin["col2"] * u.MeV,
            lat_ebin["col3"] * u.MeV - lat_ebin["col1"] * u.MeV,
        ],
        yerr=lat_ebin["col5"] * u.Unit("erg/(cm2*s)"),
        **kwargs_fp,
    )

    axs.set_ylim(
        [
            min(lat_ebin["col4"] * u.Unit("erg/(cm2*s)")) * 0.2,
            max(lat_ebin["col4"] * u.Unit("erg/(cm2*s)")) * 2,
        ]
    )

    return axs


def plot_spectrum_fold(axs=None, fold_file=None, kwargs=None):
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
