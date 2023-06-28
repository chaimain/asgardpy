import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from asgardpy.analysis import AsgardpyAnalysis
from asgardpy.config import AsgardpyConfig
from asgardpy.config.generator import CONFIG_PATH
from asgardpy.data.target import check_model_preference_aic, check_model_preference_lrt

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Get preferred best-fit spectral model")

parser.add_argument(
    "--config",
    "-c",
    help="Path to the config file",
)

# fetch options of spec models to test from user, or use all available...

parser.add_argument(
    "--write-best-config",
    help="Boolean to write the best-fit model into a separate file.",
    default=True,
    type=bool,
)


def main():
    args = parser.parse_args()

    main_config = AsgardpyConfig.read(args.config)
    target_source_name = main_config.target.source_name

    log.info(f"Analysis steps mentioned in the config file: {main_config.general.steps}")
    log.info(f"Target source is: {target_source_name}")

    spec_model_template_files = sorted(list(CONFIG_PATH.glob("model_template*yaml")))

    main_analysis_list = {}
    spec_models_list = []

    for temp in spec_model_template_files:
        temp_aa = AsgardpyAnalysis(main_config)
        temp_aa.config.target.models_file = temp
        print(temp_aa.config.target.models_file)
        temp_aa_2 = AsgardpyAnalysis(temp_aa.config)
        print(temp_aa_2.config.target)
        # Have the same value of amplitude
        temp_aa_2.config.target.components[0].spectral.parameters[0].value = (
            temp_aa.config.target.components[0].spectral.parameters[0].value
        )
        # Have the same value of reference/e_break energy
        temp_aa_2.config.target.components[0].spectral.parameters[1].value = (
            temp_aa.config.target.components[0].spectral.parameters[1].value
        )
        # Have the same value of redshift value
        temp_aa_2.config.target.components[
            0
        ].spectral.ebl_abs.redshift = temp_aa.config.target.components[0].spectral.ebl_abs.redshift

        # Make sure the source names are the same
        temp_aa_2.config.target.source_name = temp_aa.config.target.source_name
        temp_aa_2.config.target.components[0].name = temp_aa.config.target.components[0].name

        spec_tag = temp.name.split(".")[0].split("_")[-1]
        spec_models_list.append(spec_tag)
        main_analysis_list[spec_tag] = {}

        log.info(temp_aa_2.config.target.components[0])

        main_analysis_list[spec_tag]["Analysis"] = temp_aa_2

    spec_models_list = np.array(spec_models_list)

    # Run Analysis Steps till Fit
    for i, tag in enumerate(spec_models_list):
        log.info(f"Spectral model being tested: {tag}")

        main_analysis_list[tag]["Analysis"].run(["datasets-3d", "datasets-1d", "fit"])

        if tag == "pl":
            PL_idx = i

    fit_success_list = []
    pref_over_pl_chi2_list = []
    wstat_list = []
    dof_list = []

    for tag in spec_models_list:
        # Collect parameters for AIC check
        wstat = main_analysis_list[tag]["Analysis"].fit_result.total_stat
        dof = len(list(main_analysis_list[tag]["Analysis"].final_model.parameters.free_parameters))
        fit_success = main_analysis_list[tag]["Analysis"].fit_result.success

        main_analysis_list[tag]["DoF"] = dof
        main_analysis_list[tag]["Wstat"] = wstat

        fit_success_list.append(fit_success)
        wstat_list.append(wstat)
        dof_list.append(dof)

        # Checking the preference of a "nested" spectral model (observed),
        # over Power Law.
        if tag == "pl":
            main_analysis_list[tag]["Pref_over_pl_chi2"] = 0
            main_analysis_list[tag]["Pref_over_pl_pval"] = 0
            main_analysis_list[tag]["DoF_over_pl"] = 0
            pref_over_pl_chi2_list.append(0)
            continue

        p_pl_x, g_pl_x, g_pl, g_x, ndof_pl_x = check_model_preference_lrt(
            main_analysis_list["pl"]["Analysis"].fit_result,
            main_analysis_list[tag]["Analysis"].fit_result,
            main_analysis_list["pl"]["Analysis"].final_model[target_source_name],
            main_analysis_list[tag]["Analysis"].final_model[target_source_name],
        )
        log.info(f"Chi2 of {tag} model: {g_x}/{dof}")
        log.info(f"Preference of {tag} model of PL: {g_pl_x:.3f} sigmas")
        log.info(f"p-value of {tag} over PL: {p_pl_x:.3e}")

        main_analysis_list[tag]["chi2"] = g_x
        main_analysis_list["pl"]["chi2"] = g_pl
        main_analysis_list[tag]["Pref_over_pl_chi2"] = g_pl_x
        pref_over_pl_chi2_list.append(g_pl_x)
        main_analysis_list[tag]["Pref_over_pl_pval"] = p_pl_x
        main_analysis_list[tag]["DoF_over_pl"] = ndof_pl_x

    log.info(f"Chi2 of PL model: {g_pl}/{main_analysis_list['pl']['DoF']}")

    fit_success_list = np.array(fit_success_list)

    # Only select fit results that were successful for comparisons
    wstat_list = np.array(wstat_list)[fit_success_list]
    dof_list = np.array(dof_list)[fit_success_list]
    pref_over_pl_chi2_list = np.array(pref_over_pl_chi2_list)[fit_success_list]

    # If any spectral model has at least 5 sigmas preference over PL
    best_sp_idx_lrt = np.nonzero(pref_over_pl_chi2_list == np.nanmax(pref_over_pl_chi2_list))[0]
    for idx in best_sp_idx_lrt:
        if pref_over_pl_chi2_list[idx] > 5:
            sp_idx_lrt = idx
            log.info("Best preferred spectral model over PL " f"is {spec_models_list[idx]}")
        else:
            sp_idx_lrt = PL_idx
            log.info("No other model preferred over PL")

    list_rel_p = check_model_preference_aic(wstat_list, dof_list)

    for i, tag in enumerate(spec_models_list[fit_success_list]):
        log.info(f"Relative likelihood for {tag} is {list_rel_p[i]}")

        best_sp_idx_aic = np.nonzero(list_rel_p == np.nanmax(list_rel_p))[0]

        for idx in best_sp_idx_aic:
            if list_rel_p[idx] > 0.95:
                sp_idx_aic = idx
                log.info(f"Best preferred spectral model is {tag}")
            else:
                sp_idx_aic = PL_idx
                log.info("No other model preferred, hence PL is selected")

    if args.write_best_config:
        log.info("Write the spectral model")

        for idx, name in zip([sp_idx_lrt, sp_idx_aic], ["lrt", "aic"]):
            tag = spec_models_list[fit_success_list][idx]
            config_ = main_analysis_list[tag]["Analysis"].config
            spec_model = config_.target.components[0].spectral

            path = Path(args.config).parent / f"model_most_pref_{name}.yaml"
            temp_config = AsgardpyConfig()
            temp_config.target.components[0].spectral = spec_model
            temp_ = temp_config.dict(exclude_defaults=True)
            yaml_ = yaml.dump(
                temp_,
                sort_keys=False,
                indent=4,
                width=80,
                default_flow_style=None,
            )
            path.write_text(yaml_)


if __name__ == "__main__":
    main()
