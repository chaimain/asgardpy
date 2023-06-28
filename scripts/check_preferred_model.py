import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY
from scipy.stats import chi2, norm

from asgardpy.analysis import AsgardpyAnalysis
from asgardpy.config import AsgardpyConfig
from asgardpy.config.generator import CONFIG_PATH

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


def check_model_preference(result1, result2, model1, model2):
    """
    Log-likelihood ratio test. Checking the preference of a "nested" spectral
    model2 (observed), over model1.
    """
    Wstat_1 = result1.total_stat
    Wstat_2 = result2.total_stat

    p1 = chi2.sf(Wstat_1, len(list(model1.parameters.free_parameters)))
    p2 = chi2.sf(Wstat_2, len(list(model2.parameters.free_parameters)))

    g1 = norm.isf(p1 / 2)
    g2 = norm.isf(p2 / 2)

    n_dof = len(list(model2.parameters.free_parameters)) - len(
        list(model1.parameters.free_parameters)
    )
    if n_dof < 1:
        print(
            f"DoF is lower in {model2.spectral_model.model1.tag[0]} compared "
            f"to {model1.spectral_model.model1.tag[0]}"
        )
        return np.nan, np.nan, g1, g2, n_dof

    p_value = chi2.sf((Wstat_1 - Wstat_2), n_dof)
    gaussian_sigmas = norm.isf(p_value / 2)

    if not np.isfinite(gaussian_sigmas):
        gaussian_sigmas = np.sqrt((Wstat_1 - Wstat_2))

    return p_value, gaussian_sigmas, g1, g2, n_dof


def check_model_preference_aic(list_wstat, list_dof):
    """
    AIC preference over a list of wstat and DoF to get relative likelihood.
    """
    list_aic = []
    for w, d in zip(list_wstat, list_dof):
        aic = 2 * w + 2 * d
        list_aic.append(aic)
    list_aic = np.array(list_aic)

    aic_min = np.min(list_aic)
    # print(f"With a list of Akaike information criterion (AIC) Statistics,
    # {list_aic}, the minimum is {aic_min}")

    list_b = []
    for a in list_aic:
        b = np.exp((aic_min - a) / 2)
        list_b.append(b)
    list_b = np.array(list_b)

    list_p = []
    for bb in list_b:
        bbb = bb / np.sum(list_b)
        list_p.append(bbb)
    list_p = np.array(list_p)

    # print(f"Relative likelihood list: {list_p}")

    return list_p


def main():
    args = parser.parse_args()

    main_config = AsgardpyConfig().read(args.config)
    main_analysis = AsgardpyAnalysis(main_config)
    target_source_name = main_config.target.source_name

    log.info(f"Analysis steps mentioned in the config file: {main_config.general.steps}")
    log.info(f"Target source is: {target_source_name}")

    # Check if the spectral model is readable by Gammapy
    sp_t = main_analysis.config.target.components[0].spectral.type
    try:
        SPECTRAL_MODEL_REGISTRY.get_cls(sp_t)
    except KeyError:
        log.error("Incorrect spectral model that cannot be read with Gammapy")

    spec_model_template_files = sorted(list(CONFIG_PATH.parent.glob("model_template*yaml")))

    main_analysis_list = {}
    spec_models_list = []

    for temp in spec_model_template_files:
        mm = AsgardpyAnalysis(main_config)
        mm.config.target.models_file = temp
        mm_2 = AsgardpyAnalysis(mm.config)
        # Have the same value of amplitude
        mm_2.config.target.components[0].spectral.parameters[0].value = (
            mm.config.target.components[0].spectral.parameters[0].value
        )
        # Have the same value of reference/e_break energy
        mm_2.config.target.components[0].spectral.parameters[1].value = (
            mm.config.target.components[0].spectral.parameters[1].value
        )
        # Make sure the source names are the same
        mm_2.config.target.source_name = mm.config.target.source_name
        mm_2.config.target.components[0].name = mm.config.target.components[0].name

        spec_tag = SPECTRAL_MODEL_REGISTRY.get_cls(
            mm_2.config.target.components[0].spectral.type
        ).tag[1]
        spec_models_list.append(spec_tag)
        main_analysis_list[spec_tag] = {}
        main_analysis_list[spec_tag]["Analysis"] = mm_2

    spec_models_list = np.array(spec_models_list)

    # Run Analysis Steps till Fit
    for i, tag in enumerate(spec_models_list):
        main_analysis_list[tag]["Analysis"].run(["datasets-3d", "datasets-1d", "fit"])
        if tag == "pl":
            PL_idx = i

    pref_over_pl_chi2_list = []
    wstat_list = []
    dof_list = []

    for tag in spec_models_list:
        # Collect parameters for AIC check
        wstat = main_analysis_list[tag]["Analysis"].fit_result.total_stat
        dof = len(list(main_analysis_list[tag]["Analysis"].final_model.parameters.free_parameters))
        main_analysis_list[tag]["DoF"] = dof
        main_analysis_list[tag]["Wstat"] = wstat

        wstat_list.append(wstat)
        dof_list.append(dof)

        # Checking the preference of a "nested" spectral model (observed),
        # over Power Law.
        if tag == "pl":
            main_analysis_list[tag]["Pref_over_pl_chi2"] = 0
            main_analysis_list[tag]["Pref_over_pl_pval"] = 0
            main_analysis_list[tag]["DoF_over_pl"] = 0
            continue

        p_pl_x, g_pl_x, g_pl, g_x, ndof_pl_x = check_model_preference(
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

    wstat_list = np.array(wstat_list)
    dof_list = np.array(dof_list)
    pref_over_pl_chi2_list = np.array(pref_over_pl_chi2_list)

    # If any spectral model has at least 5 sigmas preference over PL
    if pref_over_pl_chi2_list.any() > 5:
        best_sp_idx_lrt = np.where(pref_over_pl_chi2_list == np.nanmax(pref_over_pl_chi2_list))
        log.info("Best preferred spectral model over PL " f"is {spec_models_list[best_sp_idx_lrt]}")
    else:
        best_sp_idx_lrt = PL_idx
        log.info("No other model preferred over PL")

    list_rel_p = check_model_preference_aic(wstat_list, dof_list)

    for i, tag in enumerate(spec_models_list):
        log.info(f"Relative likelihood for {tag} is {list_rel_p[i]}")

        if list_rel_p[i] > 0.95:
            if list_rel_p[i] == np.nanmax(list_rel_p):
                best_sp_idx_aic = i
                log.info(f"Best preferred spectral model is {tag}")
        else:
            best_sp_idx_aic = PL_idx
            log.info("No other model preferred, hence PL is selected")

    if args.write_best_config:
        log.info("Write the spectral model")

        for idx, name in zip([best_sp_idx_lrt, best_sp_idx_aic], ["lrt", "aic"]):
            tag = spec_models_list[idx]
            config_ = main_analysis_list[tag].config
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
