import pytest


@pytest.mark.test_data
def test_preferred_model(base_config_1d):
    """
    Testing the script code of checking the preferred spectral model.
    """

    import numpy as np

    from asgardpy.analysis import AsgardpyAnalysis
    from asgardpy.config.generator import CONFIG_PATH
    from asgardpy.stats.stats import (
        check_model_preference_aic,
        check_model_preference_lrt,
    )

    spec_model_template_files = sorted(list(CONFIG_PATH.glob("model_templates/model_template*yaml")))
    select_model_tags = ["lp", "bpl2", "ecpl", "pl", "eclp"]
    spec_model_temp_files = []

    for p in spec_model_template_files:
        tag = p.name.split(".")[0].split("_")[-1]

        if tag in select_model_tags:
            spec_model_temp_files.append(p)

    spec_model_temp_files = np.array(spec_model_temp_files)

    main_analysis_list = {}
    spec_models_list = []

    for temp in spec_model_temp_files:
        temp_model = AsgardpyAnalysis(base_config_1d)
        temp_model.config.fit_params.fit_range.min = "100 GeV"

        temp_model.config.target.models_file = temp

        temp_model_2 = AsgardpyAnalysis(temp_model.config)

        # Have the same value of amplitude
        temp_model_2.config.target.components[0].spectral.parameters[0].value = (
            temp_model.config.target.components[0].spectral.parameters[0].value
        )
        # Have the same value of reference/e_break energy
        temp_model_2.config.target.components[0].spectral.parameters[1].value = (
            temp_model.config.target.components[0].spectral.parameters[1].value
        )
        # Have the same value of redshift value and EBL reference model
        temp_model_2.config.target.components[0].spectral.ebl_abs.redshift = temp_model.config.target.components[
            0
        ].spectral.ebl_abs.redshift

        temp_model_2.config.target.components[0].spectral.ebl_abs.reference = temp_model.config.target.components[
            0
        ].spectral.ebl_abs.reference

        # Make sure the source names are the same
        temp_model_2.config.target.source_name = temp_model.config.target.source_name
        temp_model_2.config.target.components[0].name = temp_model.config.target.components[0].name

        spec_tag = temp.name.split(".")[0].split("_")[-1]
        spec_models_list.append(spec_tag)
        main_analysis_list[spec_tag] = {}

        main_analysis_list[spec_tag]["Analysis"] = temp_model_2

    spec_models_list = np.array(spec_models_list)

    # Run Analysis Steps till Fit
    for tag in spec_models_list:
        main_analysis_list[tag]["Analysis"].run(["datasets-1d", "fit"])

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

    # If any spectral model has at least 5 sigmas preference over PL
    best_sp_idx_lrt = np.nonzero(pref_over_pl_chi2_list == np.nanmax(pref_over_pl_chi2_list))[0]
    for idx in best_sp_idx_lrt:
        if pref_over_pl_chi2_list[idx] > 5:
            lrt_best_model = spec_models_list[idx]

    list_rel_p = check_model_preference_aic(stat_list, dof_list)

    best_sp_idx_aic = np.nonzero(list_rel_p == np.nanmax(list_rel_p))[0]

    aic_best_model = select_model_tags[best_sp_idx_aic[0]]

    assert lrt_best_model == "lp"
    assert aic_best_model == "bpl2"

    # Check for bad comparisons, same dof
    p_val_0, g_sig_0, dof_0 = check_model_preference_lrt(4.4, 2.2, 2, 2)

    assert np.isnan(p_val_0)
