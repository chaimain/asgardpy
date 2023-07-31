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
    "get_ts_target",
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


def check_model_preference_lrt(test_stat_1, test_stat_2, ndof_1, ndof_2):
    """
    Log-likelihood ratio test. Checking the preference of a "nested" spectral
    model2 (observed), over a primary model1.

    Parameters
    ----------
    test_stat_1: float
        The test statistic (-2 ln L) of the Fit result of the primary spectral model.
    test_stat_2: float
        The test statistic (-2 ln L) of the Fit result of the nested spectral model.
    ndof_1: int
        Number of degrees of freedom for the primary model
    ndof_2: int
        Number of degrees of freedom for the nested model

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


def check_model_preference_aic(list_stat, list_dof):
    """
    Akaike Information Criterion (AIC) preference over a list of stat and DoF
    (degree of freedom) to get relative likelihood of a given list of best-fit
    models.

    Parameters
    ----------
    list_wstat: list
        List of stat or -2 Log likelihood values for a list of models.
    list_dof: list
        List of degrees of freedom or list of free parameters, for a list of models.

    Returns
    -------
    list_rel_p: list
        List of relative likelihood probabilities, for a list of models.
    """
    list_aic_stat = []
    for stat, dof in zip(list_stat, list_dof):
        aic_stat = stat + 2 * dof
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


def get_goodness_of_fit_stats(datasets, instrument_spectral_info):
    """
    Evaluating the Goodness of Fit of the fitting of the model to the dataset.

    We first use the get_ts_target function to get the total test statistic of
    the null hypothesis of absence of target source signal, over all energy
    bins, for the given datasets and the fit statistics difference of the
    alternative hypothesis of the presence of the target source signal from the
    null hypothesis, summed over all energy bins for the given datasets.

    We then evaluate the total number of Degrees of Freedom for the Fit as the
    difference between the number of relevant energy bins used in the evaluation
    and the number of free model parameters.

    The fit statistics difference is used as the test statistic value for
    get_chi2_sig_pval function along with the total number of degrees of freedom
    to get the final statistics for the goodness of fit.

    The fit statistics information is updated in the dict object provided and
    a logging message is passed.

    Parameter
    ---------
    datasets: `gammapy.datasets.Datasets`
        List of Datasets object, which can contain 3D and/or 1D datasets
    instrument_spectral_info: dict
        Dict of information for storing relevant fit stats

    Return
    ------
    instrument_spectral_info: dict
        Filled Dict of information with relevant fit statistics
    stat_message: str
        String for logging the fit statistics
    """
    stat_h0, stat_ts = get_ts_target(datasets)

    instrument_spectral_info["stat_h0"] = stat_h0
    instrument_spectral_info["fit_stat"] = stat_ts
    instrument_spectral_info["stat_h1"] = stat_h0 - stat_ts
    ndof = instrument_spectral_info["DoF"]

    fit_chi2_sig, fit_pval = get_chi2_sig_pval(stat_ts, ndof)

    instrument_spectral_info["fit_chi2_sig"] = fit_chi2_sig
    instrument_spectral_info["fit_pval"] = fit_pval

    stat_message = "The Chi2/dof value of the goodness of Fit is "
    stat_message += f"{stat_ts:.2f}/{ndof}\nand the p-value is {fit_pval:.3e} "
    stat_message += f"and in Significance {fit_chi2_sig:.2f} sigmas"
    stat_message += f"\nwith TS (H0) as {stat_h0:.3f} and TS (H1) as {stat_h0-stat_ts:.3f}"

    return instrument_spectral_info, stat_message


def get_ts_target(datasets):
    """
    From a given list of DL4 datasets, with assumed associated models, estimate
    the total test statistic value for the null hypothesis that the data
    contains only background events and also the total test statistic difference
    between the null and alternative hypothesis.

    For the different type of Statistics used in Gammapy for 3D/1D datasets,
    use the following links for more details on the exact methods used for the
    calculation:

    * stat_null:

        * `Cash stat_null <https://docs.gammapy.org/1.1/api/gammapy.stats.CashCountsStatistic.html#gammapy.stats.CashCountsStatistic.stat_null # noqa>`_

        * `Wstat stat_null <https://docs.gammapy.org/1.1/api/gammapy.stats.WStatCountsStatistic.html#gammapy.stats.WStatCountsStatistic.stat_null # noqa>`_

    * ts:

        * `Cash ts <https://docs.gammapy.org/1.1/api/gammapy.stats.CashCountsStatistic.html#gammapy.stats.CashCountsStatistic.ts # noqa>`_

        * `Wstat ts <https://docs.gammapy.org/1.1/api/gammapy.stats.WStatCountsStatistic.html#gammapy.stats.WStatCountsStatistic.ts # noqa>`_

    Parameter
    ---------
    datasets: `gammapy.datasets.Datasets`
        List of Datasets object, which can contain 3D and/or 1D datasets

    Return
    ------
    stat_h0: float
        Total sum of test statistic of the null hypothesis, summed over all
        energy bins.
    stat_ts: float
        Test statistic difference of the null hypothesis and the alternative
        hypothesis (no excess vs excess) summed over all energy bins.
    """
    stat_h0 = 0
    stat_ts = 0

    for data in datasets:
        # Assuming that the Counts Map is created with the target source as its center
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

            with np.errstate(invalid="ignore", divide="ignore"):
                alpha = (
                    np.nan_to_num((data.background * data.mask).get_spectrum(region)) / counts_off
                )
            mu_signal = np.nan_to_num((data.npred_signal() * data.mask).get_spectrum(region))

            stat = WStatCountsStatistic(counts_on, counts_off, alpha, mu_signal)
        stat_h0 += np.nansum(stat.stat_null.ravel())
        stat_ts += np.nansum(stat.ts.ravel())

    return stat_h0, stat_ts
