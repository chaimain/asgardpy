import pytest

from asgardpy.analysis import AsgardpyAnalysis


@pytest.mark.test_data
def test_dataset3d(base_config):
    analysis = AsgardpyAnalysis(base_config)

    analysis.get_3d_datasets()


@pytest.mark.test_data
def test_dataset3d_different_config(base_config):
    analysis = AsgardpyAnalysis(base_config)

    analysis.config.target.from_3d = True

    analysis.get_3d_datasets


@pytest.mark.test_data
def test_only_3d_full_analysis(base_config):
    analysis = AsgardpyAnalysis(base_config)

    analysis.config.fit_params.fit_range.max = "500 GeV"

    analysis.run(["datasets-3d"])
    analysis.run_fit()
    analysis.get_flux_points()
