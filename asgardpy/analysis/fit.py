"""
Main Fit and Spectral Analysis classes
"""
import os
import warnings
import uproot
import numpy as np

import astropy.units as u
from astropy.table import Table
from astropy.units import Quantity
from gammapy.datasets import Datasets, FluxPointsDataset
from gammapy.estimators import FluxPointsEstimator, LightCurveEstimator
from gammapy.irf import EDispKernel
from gammapy.maps import Map  # , MapAxis
from gammapy.modeling import Fit
from gammapy.modeling.models import SkyModel


class FitMaker:
    """
    Basic class to setup for processes to perform the global fit.
    """
    def __init__(self, analyses, target_name, *args, **kwargs):
        self.datasets = Datasets()

        self._set_analysis_objects(analyses)
        self._setup_fit(*args, **kwargs)

        self.target_model = None
        self.fit_bin = None
        self.result = None

        self.set_target_source(target_name)

    def _set_analysis_objects(self, analyses):
        self.analyses = analyses
        self.set_datasets([A.dataset for A in self.analyses])

    def _setup_fit(self, *args, **kwargs):
        self.fit = Fit(*args, **kwargs)

    def set_target_source(self, target_name, dataset=None):
        if dataset is not None:
            dataset = [dataset]
        else:
            dataset = self.datasets

        for data in dataset:
            for mod in data.models:
                if mod.name == target_name:
                    self.target_model = mod

    def set_energy_mask(self, dataset_, en_min, en_max):
        """
        """
        coords = dataset_.counts.geom.get_coord()
        mask_energy = (coords["energy"] >= en_min) * (coords["energy"] <= en_max)
        dataset_.mask_fit = Map.from_geom(geom=dataset_.counts.geom, data=mask_energy)

        return dataset_

    def set_datasets(self, datasets, safe_en_min=None, safe_en_max=None):
        """
        """
        if safe_en_min is None:
            safe_en_min = 100 * u.MeV
        if safe_en_max is None:
            safe_en_max = 30 * u.TeV

        for data in datasets:
            data = self.set_energy_mask(
                dataset_=data,
                en_min=safe_en_min,
                en_max=safe_en_max,
            )
            self.datasets.append(data)

    # Justify
    def fit_energy_bin(self, energy_true, energy_reco, data, *args, **kwargs):
        """
        """
        warnings.filterwarnings("ignore")
        for dataset in self.datasets:
            dataset.edisp_interp_kernel = EDispKernel(
                axes=[energy_true, energy_reco], data=data
            )

        self.fit_bin = Fit(*args, **kwargs)
        self.result = self.fit_bin.run(datasets=self.datasets)
        warnings.filterwarnings("default")

        return (self.fit_bin, self.result)


class SpectralAnalysis(FitMaker):
    """
    Main class to perform the spectral analysis and make some plots.
    The plot functions may be moved in a separate location
    """

    def __init__(self, analyses, target_name, *args, **kwargs):
        super().__init__(analyses, target_name, *args, **kwargs)
        self.lat_ebin = None
        self.lat_bute = None
        self.energy_bin_edges = None
        self.flux_points = None
        self.flux_points_dataset = None
        self.light_curve_flux = None

    def read_enrico_spectrum(self, lat_ebin_file=None, lat_bute_file=None):
        """
        Retrieves the Fit energy edges used in creating the enrico LAT spectrum
        """
        if lat_ebin_file is None:
            lat_ebin_file = self.analyses[0].lat_ebin_file[0]
        if lat_bute_file is None:
            lat_bute_file = self.analyses[0].lat_bute_file[0]

        self.lat_ebin = Table.read(lat_ebin_file, format="ascii")
        self.lat_bute = Table.read(lat_bute_file, format="ascii")
        self.energy_bin_edges = (
            np.append(self.lat_ebin["col2"][0], self.lat_ebin["col3"]) * u.MeV
        )

    """
    def prepare_energy_bins(self, dataset, energy_bin_edges=None):

        # Using the Energy Dispersion Matrix IRF included with the dataset, to
        # get the Fit or Reco energy bins.

        energy_true_axis = dataset.edisp_interp_kernel.axes[0]

        for k, ebin_lo in energy_bin_edges[0:-1]:
            ebin_hi = energy_bin_edges[k + 1]
            energy_true_slice = MapAxis.from_energy_edges(
                np.append(ebin_lo, ebin_hi)
            )

            for dataset in self.datasets:
                ## Explanation/Description required
                energy_reco_axis_slice, jmin, jmax = slice_in_mapaxis(
                    energy_reco_axis, ebin_lo, ebin_hi, 2
                )
                # energy_true_axis_slice,imin,imax = slice_in_mapaxis(energy_true_axis,ebin_lo,ebin_hi,0)

                drm_interp = dataset.edisp_interp_kernel.valuate(
                    **{"energy": energy_reco_axis, "energy_true": energy_true_slice}
                )

                dataset.edisp_interp_kernel = EDispKernel(
                    axes=[axis_true, axis_reco], data=np.asarray(drm_interp)
                )

            self.fit_energy_bin()
    """

    def global_fit(self, datasets=None):
        """
        """
        warnings.filterwarnings("ignore")

        if datasets is None:
            datasets = self.datasets
        self.result = self.fit.run(datasets=datasets)
        warnings.filterwarnings("default")

    def print_parameters(self, only_first_dataset=True, full_datasets=False):
        """
        """
        if only_first_dataset:
            datasets = [
                self.datasets[0],
            ]
        else:
            datasets = self.datasets

        for data in datasets:
            if full_datasets:
                data.models.to_parameters_table().pprint_all()
            else:
                data.models.to_parameters_table().pprint()

    def get_spectral_points(
        self,
        energy_bin_edges=None,
        target_name=None,
        datasets=None
    ):
        """
        """
        warnings.filterwarnings("ignore")

        if datasets is None:
            datasets = self.datasets

        if energy_bin_edges is not None:
            self.energy_bin_edges = energy_bin_edges

        if target_name is None:
            target_name = self.analyses[0].target_name

        fpe = FluxPointsEstimator(
            energy_edges=self.energy_bin_edges,
            source=self.target_model.name,
            n_sigma_ul=2,
            selection_optional="all",
        )

        self.flux_points = fpe.run(datasets=datasets)
        warnings.filterwarnings("default")

    def get_lc_flux_points(
        self,
        energy_range=None,
        time_intervals=None,
        target_name=None,
        datasets=None,
        reoptimize=False
    ):
        """
        """
        warnings.filterwarnings("ignore")

        if datasets is None:
            datasets = self.datasets

        if energy_range is not None:
            energy_range = Quantity([self.energy_bin_edges[0], self.energy_bin_edges[-1]])

        if target_name is None:
            target_name = self.analyses[0].target_name

        lc_flux = LightCurveEstimator(
            energy_edges=energy_range,
            time_intervals=time_intervals,
            source=self.target_model.name,
            reoptimize=reoptimize,
            selection_optional="all",
        )

        self.light_curve_flux = lc_flux.run(datasets=datasets)
        warnings.filterwarnings("default")

    def plot_parameter_stat_profile(self, datasets, parameter, axs=None):
        """
        """
        total_stat = self.result.total_stat

        parameter.scan_n_values = 20

        profile = self.fit.stat_profile(datasets=datasets, parameter=parameter)

        axs.plot(
            profile[f"{parameter.name}_scan"],
            profile["stat_scan"] - total_stat
        )
        axs.set_xlabel(f"{parameter.unit}")
        axs.set_ylabel("Delta TS")
        axs.set_title(
            f"{parameter.name}: {parameter.value:.2e} +/- {parameter.error:.2e}"
            + f"\n{parameter.value:.2e}+/- {parameter.error/parameter.value:.2f}"
        )

        return axs

    def plot_spectrum_fp(self, axs=None, kwargs_fp=None):
        """
        """
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
        """
        """
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
        """
        """
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

            spec.plot_error(
                ax=axs, energy_bounds=energy_range, **kwargs_model_err
            )
            spec.plot(ax=axs, energy_bounds=energy_range, **kwargs_model)

        return axs

    def plot_residuals(self, axs=None, method="diff", kwargs_res=None):
        """
        """
        self.flux_points_dataset = FluxPointsDataset(
            data=self.flux_points,
            models=SkyModel(spectral_model=self.target_model.spectral_model)
        )

        self.flux_points_dataset.plot_residuals(ax=axs, method=method, **kwargs_res)

        return axs

    def plot_ts_profiles(self, axs=None, add_cbar=True, kwargs_ts=None):
        """
        """
        if kwargs_ts is None:
            kwargs_ts = {
                "sed_type": "e2dnde",
                "color": "darkorange",
            }
        self.flux_points.plot_ts_profiles(ax=axs, add_cbar=add_cbar, **kwargs_ts)
        self.flux_points.plot(ax=axs, **kwargs_ts)

        return axs

    def plot_model_covariance_correlation(self, axs=None):
        """
        """
        spec = self.target_model.spectral_model
        spec.covariance.plot_correlation(ax=axs)

        return axs

    def plot_spectrum_enrico(
        self, axs=None, kwargs_fp=None, kwargs_model=None, kwargs_model_err=None
    ):
        """
        """
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
        """
        """
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
            yerr=[
                y_err_low * u.Unit("TeV/(cm2 * s)"),
                y_err_high * u.Unit("TeV/(cm2 * s)"),
            ],
            **kwargs,
        )

        return axs
