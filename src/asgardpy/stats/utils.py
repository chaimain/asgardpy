"""
Module containing additional utility functions for selecting a preferred model.
"""

import numpy as np
from astropy.table import QTable

from asgardpy.config.operations import all_model_templates
from asgardpy.stats.stats import check_model_preference_lrt

__all__ = [
    "fetch_all_analysis_fit_info",
    "get_model_config_files",
    "tabulate_best_fit_stats",
    "copy_target_config",
]


def get_model_config_files(select_model_tags):
    """From the default model templates, select some."""

    all_tags, template_files = all_model_templates()

    spec_model_temp_files = []
    for tag in select_model_tags:
        spec_model_temp_files.append(template_files[np.where(all_tags == tag)[0][0]])

    spec_model_temp_files = np.array(spec_model_temp_files)

    return spec_model_temp_files


def get_spec_params_indices(aa_config):
    """
    For copying the spectral flux amplitude and flux normalization energy,
    from one config to another, find the correct parameter indices within a
    given config.
    """
    par_names = []
    for p in aa_config.config.target.components[0].spectral.parameters:
        par_names.append(p.name)
    par_names = np.array(par_names)

    amp_idx = None
    # For models without this parameter, name has not yet been included or
    # checked with Asgardpy
    if "amplitude" in par_names:
        amp_idx = np.where(par_names == "amplitude")[0][0]

    if "reference" in par_names:
        eref_idx = np.where(par_names == "reference")[0][0]
    else:
        eref_idx = np.where(par_names == "ebreak")[0][0]

    return amp_idx, eref_idx


def copy_target_config(aa_config_1, aa_config_2):
    """From aa_config_1 update information in aa_config_2."""

    amp_idx_1, eref_idx_1 = get_spec_params_indices(aa_config_1)
    amp_idx_2, eref_idx_2 = get_spec_params_indices(aa_config_2)

    # Have the same value of amplitude
    aa_config_2.config.target.components[0].spectral.parameters[amp_idx_2].value = (
        aa_config_1.config.target.components[0].spectral.parameters[amp_idx_1].value
    )
    # Have the same value of reference/e_break energy
    aa_config_2.config.target.components[0].spectral.parameters[eref_idx_2].value = (
        aa_config_1.config.target.components[0].spectral.parameters[eref_idx_1].value
    )
    # Have the same value of redshift value and EBL reference model
    aa_config_2.config.target.components[0].spectral.ebl_abs.redshift = aa_config_1.config.target.components[
        0
    ].spectral.ebl_abs.redshift

    # Make sure the source names are the same
    aa_config_2.config.target.source_name = aa_config_1.config.target.source_name
    aa_config_2.config.target.components[0].name = aa_config_1.config.target.components[0].name

    return aa_config_2


def fetch_all_analysis_fit_info(main_analysis_list, spec_models_list):
    """
    For a list of spectral models, with the AsgardpyAnalysis run till the fit
    step, get the relevant information for testing the model preference.
    """
    fit_success_list = []
    pref_over_pl_chi2_list = []
    stat_list = []
    dof_list = []

    for tag in spec_models_list:
        dict_tag = main_analysis_list[tag]["Analysis"].instrument_spectral_info
        dict_pl = main_analysis_list["pl"]["Analysis"].instrument_spectral_info

        # Collect parameters for AIC check
        stat = dict_tag["best_fit_stat"]
        dof = dict_tag["DoF"]

        fit_success = main_analysis_list[tag]["Analysis"].fit_result.success

        fit_success_list.append(fit_success)
        stat_list.append(stat)
        dof_list.append(dof)

        # Checking the preference of a "nested" spectral model (observed),
        # over Power Law.
        if tag == "pl":
            main_analysis_list[tag]["Pref_over_pl_chi2"] = 0
            main_analysis_list[tag]["Pref_over_pl_pval"] = 0
            main_analysis_list[tag]["DoF_over_pl"] = 0
            pref_over_pl_chi2_list.append(0)
            continue

        p_pl_x, g_pl_x, ndof_pl_x = check_model_preference_lrt(
            dict_pl["best_fit_stat"],
            dict_tag["best_fit_stat"],
            dict_pl["DoF"],
            dict_tag["DoF"],
        )

        main_analysis_list[tag]["Pref_over_pl_chi2"] = g_pl_x
        pref_over_pl_chi2_list.append(g_pl_x)
        main_analysis_list[tag]["Pref_over_pl_pval"] = p_pl_x
        main_analysis_list[tag]["DoF_over_pl"] = ndof_pl_x

    fit_success_list = np.array(fit_success_list)

    # Only select fit results that were successful for comparisons
    stat_list = np.array(stat_list)[fit_success_list]
    dof_list = np.array(dof_list)[fit_success_list]
    pref_over_pl_chi2_list = np.array(pref_over_pl_chi2_list)[fit_success_list]

    return fit_success_list, stat_list, dof_list, pref_over_pl_chi2_list


def tabulate_best_fit_stats(spec_models_list, fit_success_list, main_analysis_list, list_rel_p):
    """For a list of spectral models, tabulate the best fit information."""

    fit_stats_table = []

    for i, tag in enumerate(spec_models_list[fit_success_list]):
        info_ = main_analysis_list[tag]["Analysis"].instrument_spectral_info

        t = main_analysis_list[tag]

        ts_gof = round(info_["best_fit_stat"] - info_["max_fit_stat"], 3)
        t_fits = {
            "Spectral Model": tag.upper(),
            "TS of Best Fit": round(info_["best_fit_stat"], 3),
            "TS of Max Fit": round(info_["max_fit_stat"], 3),
            "TS of Goodness of Fit": ts_gof,
            "DoF of Fit": info_["DoF"],
            r"Significance ($\sigma$) of Goodness of Fit": round(info_["fit_chi2_sig"], 3),
            "p-value of Goodness of Fit": float(f"{info_['fit_pval']:.4g}"),
            "Pref over PL (chi2)": round(t["Pref_over_pl_chi2"], 3),
            "Pref over PL (p-value)": float(f"{t['Pref_over_pl_pval']:.4g}"),
            "Pref over PL (DoF)": t["DoF_over_pl"],
            "Relative p-value (AIC)": float(f"{list_rel_p[i]:.4g}"),
        }
        fit_stats_table.append(t_fits)
    stats_table = QTable(fit_stats_table)

    return stats_table
