import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from astropy.table import QTable

from asgardpy.analysis import AsgardpyAnalysis
from asgardpy.config import AsgardpyConfig
from asgardpy.config.generator import CONFIG_PATH
from asgardpy.stats.stats import (
    check_model_preference_aic,
    check_model_preference_lrt,
)

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Get preferred best-fit spectral model")

parser.add_argument(
    "--config",
    "-c",
    help="Path to the config file",
)

# fetch options of spec models to test from user, or use all available...
parser.add_argument(
    "--ebl-scale-factor", "-e", help="Value of EBL Norm Scale Factor", default=1.0, type=float
)

parser.add_argument(
    "--ebl-model-name",
    "-m",
    help="Name of EBL model as used by Gammapy",
    default="dominguez",
    type=str,
)

parser.add_argument(
    "--write-config",
    help="Boolean to write the best-fit model into a separate file.",
    default=True,
    type=bool,
)


def main():
    args = parser.parse_args()

    main_config = AsgardpyConfig.read(args.config)
    config_path = Path(args.config)
    config_path_file_name = config_path.name.split(".")[0]
    target_source_name = main_config.target.source_name

    steps_list = []
    for s in main_config.general.steps:
        if s.value != "flux-points":
            steps_list.append(s.value)
    log.info("Target source is: %s", target_source_name)

    spec_model_template_files = sorted(list(CONFIG_PATH.glob("model_template*yaml")))

    # Only use the following spectral model templates for comparison
    select_model_tags = ["lp", "bpl", "ecpl", "pl", "eclp", "sbpl"]
    spec_model_temp_files = []

    for p in spec_model_template_files:
        tag = p.name.split(".")[0].split("_")[-1]

        if tag in select_model_tags:
            spec_model_temp_files.append(p)

    spec_model_temp_files = np.array(spec_model_temp_files)

    main_analysis_list = {}
    spec_models_list = []

    for temp in spec_model_temp_files:
        temp_model = AsgardpyAnalysis(main_config)
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
        temp_model_2.config.target.components[
            0
        ].spectral.ebl_abs.redshift = temp_model.config.target.components[
            0
        ].spectral.ebl_abs.redshift

        if args.ebl_scale_factor != 1.0:
            temp_model_2.config.target.components[
                0
            ].spectral.ebl_abs.alpha_norm = args.ebl_scale_factor

        if args.ebl_model_name != "dominguez":
            temp_model_2.config.target.components[
                0
            ].spectral.ebl_abs.reference = args.ebl_model_name.replace("_", "-")
        else:
            temp_model_2.config.target.components[
                0
            ].spectral.ebl_abs.reference = temp_model.config.target.components[
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
    for i, tag in enumerate(spec_models_list):
        log.info("Spectral model being tested: %s", tag)

        main_analysis_list[tag]["Analysis"].run(steps_list)

        if tag == "pl":
            PL_idx = i

    fit_success_list = []
    pref_over_pl_chi2_list = []
    stat_list = []
    dof_list = []

    for tag in spec_models_list:
        dict_tag = main_analysis_list[tag]["Analysis"].instrument_spectral_info
        dict_pl = main_analysis_list["pl"]["Analysis"].instrument_spectral_info

        # Collect parameters for AIC check
        stat = dict_tag["fit_stat"]
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
            dict_pl["fit_stat"],
            dict_tag["fit_stat"],
            dict_tag["DoF"],
            dict_pl["DoF"],
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
            sp_idx_lrt = idx
            log.info("Best preferred spectral model over PL is %s", spec_models_list[idx])
        else:
            sp_idx_lrt = PL_idx
            log.info("No other model preferred over PL")

    list_rel_p = check_model_preference_aic(stat_list, dof_list)

    for i, tag in enumerate(spec_models_list[fit_success_list]):
        best_sp_idx_aic = np.nonzero(list_rel_p == np.nanmax(list_rel_p))[0]

        for idx in best_sp_idx_aic:
            if list_rel_p[idx] > 0.95:
                sp_idx_aic = idx
                log.info("Best preferred spectral model is %s", tag)
            else:
                sp_idx_aic = PL_idx
                log.info("No other model preferred, hence PL is selected")

    fit_stats_table = []
    for i, tag in enumerate(spec_models_list):
        info_ = main_analysis_list[tag]["Analysis"].instrument_spectral_info

        t = main_analysis_list[tag]

        t_fits = {
            "Spectral Model": tag.upper(),
            "TS of null hypothesis": round(info_["stat_null"], 3),
            "TS of Goodness of Fit": round(info_["fit_stat"], 3),
            "DoF of Fit": info_["DoF"],
            r"Significance ($\sigma$) of Goodness of Fit": round(info_["fit_chi2_sig"], 3),
            "p-value of Goodness of Fit": float("%.4g" % info_["fit_pval"]),
            "Pref over PL (chi2)": round(t["Pref_over_pl_chi2"], 3),
            "Pref over PL (p-value)": float("%.4g" % t["Pref_over_pl_pval"]),
            "Pref over PL (DoF)": t["DoF_over_pl"],
            "Relative p-value (AIC)": float("%.4g" % list_rel_p[i]),
        }
        fit_stats_table.append(t_fits)
    stats_table = QTable(fit_stats_table)

    stats_table.meta["Target source name"] = target_source_name
    stats_table.meta["EBL model"] = args.ebl_model_name
    stats_table.meta["EBL scale factor"] = args.ebl_scale_factor

    file_name = (
        f"{config_path_file_name}_{args.ebl_model_name}_{args.ebl_scale_factor}_fit_stats.ecsv"
    )
    stats_table.write(
        main_config.general.outdir / file_name,
        format="ascii.ecsv",
        overwrite=True,
    )

    if args.write_config:
        log.info("Write the spectral model")

        for idx, name in zip([sp_idx_lrt, sp_idx_aic], ["lrt", "aic"]):
            tag = spec_models_list[fit_success_list][idx]
            model_ = main_analysis_list[tag]["Analysis"].final_model[0]
            spec_model = model_.spectral_model.model1.to_dict()

            path = config_path.parent / f"{config_path_file_name}_model_most_pref_{name}.yaml"
            # Create a temp config file
            temp_config = AsgardpyConfig()
            temp_config.target.components[0] = spec_model
            # Update with the spectral model info
            temp_ = temp_config.dict(exclude_defaults=True)

            # Remove some of the unnecessary keys
            temp_["target"].pop("models_file", None)
            temp_["target"]["components"][0]["spectral"].pop("ebl_abs", None)

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
