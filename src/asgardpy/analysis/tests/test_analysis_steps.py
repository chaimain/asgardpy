import pytest

from asgardpy.analysis import AsgardpyAnalysis


@pytest.mark.test_data
def test_joint_3d_1d(base_config):
    """
    Test to run a major Fermi (3D) + HESS (1D) + MAGIC (1D) joint analysis.
    """

    analysis = AsgardpyAnalysis(base_config)

    extra_config = base_config.copy()
    extra_config.general.n_jobs = 33
    analysis.update_config(extra_config)

    analysis.run(["datasets-3d"])
    analysis.run(["datasets-1d"])
    analysis.run_fit()

    assert analysis.config.general.n_jobs == 33
    assert len(analysis.datasets) == 4
    assert len(analysis.models) == 11
    assert analysis.fit_result.success is True


@pytest.mark.test_data
def test_analysis_basics(gammapy_data_path, base_config):
    """Testing some basic analysis functions."""

    other_config_path_1 = f"{gammapy_data_path}fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml"

    base_config.target.models_file = other_config_path_1

    analysis_1 = AsgardpyAnalysis(base_config)

    with pytest.raises(RuntimeError):
        print(analysis_1.models)

    spec_model_name = analysis_1.config.target.components[0].spectral.type

    assert spec_model_name == "LogParabolaSpectralModel"

    other_config_path_2 = f"{gammapy_data_path}fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml"

    base_config.target.models_file = other_config_path_2

    with pytest.raises(TypeError):
        AsgardpyAnalysis(base_config)

    config_dict = {"general": {"n_jobs": 111}}
    analysis_1.config = config_dict

    assert analysis_1.config.general.n_jobs == 111


@pytest.mark.test_data
def test_ebl_deabsorbed(gammapy_data_path, ebl_hess_pks):
    """Testing generation of EBL-deabsorbed Flux points."""

    analysis = AsgardpyAnalysis(ebl_hess_pks)

    analysis.run()

    analysis.get_correct_ebl_deabs_flux_points()

    assert len(analysis.model_deabs.parameters) == 3
    assert analysis.flux_points_deabs
