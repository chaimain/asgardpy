import argparse
import logging
from pathlib import Path

import numpy as np

from asgardpy.analysis import AsgardpyAnalysis
from asgardpy.config import AsgardpyConfig, write_asgardpy_model_to_file
from asgardpy.stats import (
    check_model_preference_aic,
    copy_target_config,
    fetch_all_analysis_fit_info,
    get_model_config_files,
    tabulate_best_fit_stats,
)

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Get preferred best-fit spectral model")

parser.add_argument(
    "--config",
    "-c",
    help="Path to the config file",
)

parser.add_argument("--ebl-scale-factor", "-e", help="Value of EBL Norm Scale Factor", default=1.0, type=float)

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


def fetch_all_analysis_objects(main_config, spec_model_temp_files, ebl_scale_factor, ebl_model_name):
    """For a list of spectral models, initiate AsgardpyAnalysis objects."""
    main_analysis_list = {}
    spec_models_list = []

    for temp in spec_model_temp_files:
        temp_model = AsgardpyAnalysis(main_config)
        temp_model.config.target.models_file = temp

        temp_model_2 = AsgardpyAnalysis(temp_model.config)

        copy_target_config(temp_model, temp_model_2)

        if ebl_scale_factor != 1.0:
            temp_model_2.config.target.components[0].spectral.ebl_abs.alpha_norm = ebl_scale_factor

        if ebl_model_name != "dominguez":
            temp_model_2.config.target.components[0].spectral.ebl_abs.reference = ebl_model_name.replace("_", "-")
        else:
            temp_model_2.config.target.components[
                0
            ].spectral.ebl_abs.reference = temp_model.config.target.components[0].spectral.ebl_abs.reference

        spec_tag = temp.name.split(".")[0].split("_")[-1]
        spec_models_list.append(spec_tag)
        main_analysis_list[spec_tag] = {}

        main_analysis_list[spec_tag]["Analysis"] = temp_model_2

    spec_models_list = np.array(spec_models_list)

    return main_analysis_list, spec_models_list


def main():
    args = parser.parse_args()

    main_config = AsgardpyConfig.read(args.config)
    config_path = Path(args.config)
    config_path_file_name = config_path.name.split(".")[0]
    target_source_name = main_config.target.source_name

    steps_list = []
    for s in main_config.general.steps:
        if s != "flux-points":
            steps_list.append(s)
    log.info("Target source is: %s", target_source_name)

    spec_model_temp_files = get_model_config_files(["lp", "bpl", "ecpl", "pl", "eclp", "sbpl"])

    main_analysis_list, spec_models_list = fetch_all_analysis_objects(
        main_config, spec_model_temp_files, args.ebl_scale_factor, args.ebl_model_name
    )

    # Run Analysis Steps till Fit
    PL_idx = 0

    for i, tag in enumerate(spec_models_list):
        log.info("Spectral model being tested: %s", tag)

        main_analysis_list[tag]["Analysis"].run(steps_list)

        if tag == "pl":
            PL_idx = i

    fit_success_list, stat_list, dof_list, pref_over_pl_chi2_list = fetch_all_analysis_fit_info(
        main_analysis_list, spec_models_list
    )

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

    best_sp_idx_aic = np.nonzero(list_rel_p == np.nanmax(list_rel_p))[0]

    for idx in best_sp_idx_aic:
        if list_rel_p[idx] > 0.95:
            sp_idx_aic = idx
            log.info("Best preferred spectral model is %s", spec_models_list[fit_success_list][idx])
        else:
            sp_idx_aic = PL_idx
            log.info("No other model preferred, hence PL is selected")

    stats_table = tabulate_best_fit_stats(spec_models_list, fit_success_list, main_analysis_list, list_rel_p)

    stats_table.meta["Target source name"] = target_source_name
    stats_table.meta["EBL model"] = args.ebl_model_name
    stats_table.meta["EBL scale factor"] = args.ebl_scale_factor

    file_name = f"{config_path_file_name}_{args.ebl_model_name}_{args.ebl_scale_factor}_fit_stats.ecsv"
    stats_table.write(
        main_config.general.outdir / file_name,
        format="ascii.ecsv",
        overwrite=True,
    )

    if args.write_config:
        log.info("Write the spectral model")

        for idx, name in zip([sp_idx_lrt, sp_idx_aic], ["lrt", "aic"], strict=False):
            tag = spec_models_list[fit_success_list][idx]

            path = config_path.parent / f"{config_path_file_name}_model_most_pref_{name}.yaml"

            write_asgardpy_model_to_file(
                gammapy_model=main_analysis_list[tag]["Analysis"].final_model[0], output_file=path
            )


if __name__ == "__main__":
    main()
