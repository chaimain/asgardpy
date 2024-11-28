import numpy as np

from asgardpy.analysis import AsgardpyAnalysis
from asgardpy.stats import (
    check_model_preference_aic,
    check_model_preference_lrt,
    copy_target_config,
    fetch_all_analysis_fit_info,
    get_model_config_files,
    tabulate_best_fit_stats,
)


def test_preferred_model(base_config_1d):
    """
    Testing the script code of checking the preferred spectral model.
    """
    select_model_tags = ["lp", "bpl2", "ecpl", "pl", "eclp"]
    spec_model_temp_files = []
    spec_model_temp_files = get_model_config_files(select_model_tags)

    main_analysis_list = {}
    spec_models_list = []

    for temp in spec_model_temp_files:
        temp_model = AsgardpyAnalysis(base_config_1d)
        temp_model.config.fit_params.fit_range.min = "100 GeV"

        temp_model.config.target.models_file = temp

        temp_model_2 = AsgardpyAnalysis(temp_model.config)

        copy_target_config(temp_model, temp_model_2)

        spec_tag = temp.name.split(".")[0].split("_")[-1]
        spec_models_list.append(spec_tag)
        main_analysis_list[spec_tag] = {}

        main_analysis_list[spec_tag]["Analysis"] = temp_model_2

    spec_models_list = np.array(spec_models_list)

    # Run Analysis Steps till Fit
    for tag in spec_models_list:
        main_analysis_list[tag]["Analysis"].run(["datasets-1d", "fit"])

    fit_success_list, stat_list, dof_list, pref_over_pl_chi2_list = fetch_all_analysis_fit_info(
        main_analysis_list, spec_models_list
    )

    # If any spectral model has at least 5 sigmas preference over PL
    best_sp_idx_lrt = np.nonzero(pref_over_pl_chi2_list == np.nanmax(pref_over_pl_chi2_list))[0]
    for idx in best_sp_idx_lrt:
        if pref_over_pl_chi2_list[idx] > 5:
            lrt_best_model = spec_models_list[idx]

    list_rel_p = check_model_preference_aic(stat_list, dof_list)

    best_sp_idx_aic = np.nonzero(list_rel_p == np.nanmax(list_rel_p))[0]

    aic_best_model = select_model_tags[best_sp_idx_aic[0]]

    stats_table = tabulate_best_fit_stats(spec_models_list, fit_success_list, main_analysis_list, list_rel_p)

    assert lrt_best_model == "lp"
    assert aic_best_model == "lp"
    assert len(stats_table.colnames) == 11

    # Check for bad comparisons, same dof
    p_val_0, g_sig_0, dof_0 = check_model_preference_lrt(4.4, 2.2, 2, 2)

    assert np.isnan(p_val_0)
