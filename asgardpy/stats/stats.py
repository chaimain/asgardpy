"""
Module for performing some statistic functions.
"""
import numpy as np
from gammapy.stats import CashCountsStatistic, WStatCountsStatistic
from scipy.stats import chi2, norm

__all__ = [
    "check_model_preference_aic",
    "check_model_preference_lrt",
    "get_chi2_sig_pval",
    "get_goodness_of_fit_stats",
    "get_ts_null_hypothesis",
]


def get_chi2_sig_pval(test_stat, ndof):
    """
    Using the log-likelihood value for a model fitting to data, along with the
    total degrees of freedom, evaluate the significance value in terms of gaussian
    distribution along with one-tailed p-value for the fitting statistics.

    In Gammapy, for 3D analysis, cash statistics is used, while for 1D analysis,
    wstat statistics is used. Check the documentation for more details
    https://docs.gammapy.org/1.1/user-guide/stats/index.html

    Parameters
    ----------
    test_stat: float
        The test statistic (-2 ln L) value of the fitting.
    ndof: int
        Total number of degrees of freedom.

    Returns
    -------
    chi2_sig: float
        significance (Chi2) of the likelihood of fit model estimated in
        Gaussian distribution.
    pval: float
        p-value for the model fitting

    """
    pval = chi2.sf(test_stat, ndof)
    chi2_sig = norm.isf(pval / 2)

    if not np.isfinite(chi2_sig):
        chi2_sig = np.sqrt(test_stat)

    return chi2_sig, pval


def get_goodness_of_fit_stats(instrument_spectral_info):
    """
    Evaluating the Goodness of Fit of the fitting of the model to the data.

    This is done by using the total TS of the null hypothesis for each dataset
    used, as TS_H0, and the optimized fit statistics, as TS_H1, the alternate
    hypothesis of the presence of target source with signal (data with the best
    fit model), to get the final input of test_statistic for get_chi2_sig_pval
    function to be, sqrt(TS_H0 - TS_H1).

    The Degrees of Freedom for the Fit is taken as (the number of
    relevant energy bins used in the evaluation) - (the number of
    free model parameters).

    Parameter
    ---------
    instrument_spectral_info: dict
        Dict of information for storing relevant fit stats

    Return
    ------
    instrument_spectral_info: dict
        Filled Dict of information with relevant fit statistics
    stat_message: str
        String for logging the fit statistics
    """
    stat = np.sqrt(instrument_spectral_info["TS_H0"] - instrument_spectral_info["TS_H1"])
    ndof = instrument_spectral_info["DoF"]
    fit_chi2_sig, fit_pval = get_chi2_sig_pval(stat, ndof)

    instrument_spectral_info["fit_stat"] = stat
    instrument_spectral_info["fit_chi2_sig"] = fit_chi2_sig
    instrument_spectral_info["fit_pval"] = fit_pval

    stat_message = f"The Chi2/dof value of the goodness of Fit is {stat**2:.2f}/{ndof}"
    stat_message += f"\nand the p-value is {fit_pval:.3e} and in "
    stat_message += f"Significance {fit_chi2_sig:.2f} sigmas"

    return instrument_spectral_info, stat_message


def check_model_preference_lrt(test_stat_1, test_stat_2, ndof_1, ndof_2):
    """
    Log-likelihood ratio test. Checking the preference of a "nested" spectral
    model2 (observed), over a primary model1.

    Parameters
    ----------
    test_stat_1: `gammapy.modeling.fit.FitResult.total_stat`
        The test statistic (-2 ln L) of the Fit result of the primary spectral model.
    test_stat_2: `gammapy.modeling.fit.FitResult.total_stat`
        The test statistic (-2 ln L) of the Fit result of the nested spectral model.
    ndof_1: 'int'
        Number of energy bins used for spectral fit - number of free spectral parameters for the primary model
    ndof_2: 'int'
        Number of energy bins used for spectral fit - number of free spectral parameters for the nested model

    Returns
    -------
    p_value: float
        p-value for the ratio of the likelihoods
    gaussian_sigmas: float
        significance (Chi2) of the ratio of the likelihoods estimated in
        Gaussian distribution.
    n_dof: int
        number of degrees of freedom or free parameters between primary and
        nested model.
    """
    n_dof = ndof_1 - ndof_2

    if n_dof < 1:
        print(f"DoF is lower in {ndof_1} compared to {ndof_2}")

        return np.nan, np.nan, n_dof

    gaussian_sigmas, p_value = get_chi2_sig_pval(test_stat_1 - test_stat_2, n_dof)

    return p_value, gaussian_sigmas, n_dof


def check_model_preference_aic(list_wstat, list_dof):
    """
    Akaike Information Criterion (AIC) preference over a list of wstat and DoF
    (degree of freedom) to get relative likelihood of a given list of best-fit
    models.

    Parameters
    ----------
    list_wstat: list
        List of wstat or -2 Log likelihood values for a list of models.
    list_dof: list
        List of degrees of freedom or list of free parameters, for a list of models.

    Returns
    -------
    list_rel_p: list
        List of relative likelihood probabilities, for a list of models.
    """
    list_aic_stat = []
    for wstat, dof in zip(list_wstat, list_dof):
        aic_stat = wstat + 2 * dof
        list_aic_stat.append(aic_stat)
    list_aic_stat = np.array(list_aic_stat)

    aic_stat_min = np.min(list_aic_stat)

    list_b_stat = []
    for aic in list_aic_stat:
        b_stat = np.exp((aic_stat_min - aic) / 2)
        list_b_stat.append(b_stat)
    list_b_stat = np.array(list_b_stat)

    list_rel_p = []
    for b_stat in list_b_stat:
        rel_p = b_stat / np.sum(list_b_stat)
        list_rel_p.append(rel_p)
    list_rel_p = np.array(list_rel_p)

    return list_rel_p


def get_ts_null_hypothesis(datasets):
    """
    From a given list of DL4 datasets, with assumed associated models, estimate
    the total test statistic value for the null hypothesis that the data
    contains only background events.

    Parameter
    ---------
    datasets: `gammapy.datasets.Datasets`
        List of Datasets object, which can contain 3D and/or 1D datasets

    Return
    ------
    ts_h0: float
        Total sum of test statistic of the null hypothesis
    """
    ts_h0 = 0

    for data in datasets:
        region = data.counts.geom.center_skydir

        if data.stat_type == "cash":
            counts_on = (data.counts.copy() * data.mask).get_spectrum(region).data
            mu_on = (data.npred() * data.mask).get_spectrum(region).data

            stat = CashCountsStatistic(
                counts_on,
                mu_on,
            )
        elif data.stat_type == "wstat":
            counts_on = (data.counts.copy() * data.mask).get_spectrum(region)
            counts_off = np.nan_to_num((data.counts_off * data.mask).get_spectrum(region))
            alpha = np.nan_to_num((data.background * data.mask).get_spectrum(region)) / counts_off
            mu_signal = np.nan_to_num((data.npred_signal() * data.mask).get_spectrum(region))

            stat = WStatCountsStatistic(counts_on, counts_off, alpha, mu_signal)
        ts_h0 += np.nansum(stat.stat_null.ravel())

    return ts_h0
