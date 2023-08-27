import pytest

from asgardpy.analysis import AsgardpyAnalysis


@pytest.mark.test_data
def test_fit_1d(base_config_1d):
    analysis = AsgardpyAnalysis(base_config_1d)

    analysis.run(["datasets-1d", "fit"])


"""
@pytest.mark.test_data
def test_fit_3d(base_config):
    analysis = AsgardpyAnalysis(base_config)

    analysis.config.fit_params.fit_range.max = "562.34132519 GeV"

    analysis.run(["datasets-3d", "fit"])


@pytest.mark.test_data
def test_fit_1d_3d(base_config):
    analysis = AsgardpyAnalysis(base_config)

    analysis.run(["datasets-3d", "datasets-1d", "fit"])


@pytest.mark.test_data
def test_flux_points_1d(base_config_1d):
    analysis = AsgardpyAnalysis(base_config_1d)

    analysis.run(["datasets-1d", "fit", "flux-points"])


@pytest.mark.test_data
def test_flux_points_3d(base_config):
    analysis = AsgardpyAnalysis(base_config)

    analysis.config.fit_params.fit_range.max = "562.34132519 GeV"

    analysis.run(["datasets-3d", "fit", "flux-points"])


@pytest.mark.test_data
def test_flux_points_1d_3d(base_config):
    analysis = AsgardpyAnalysis(base_config)

    analysis.run(["datasets-3d", "datasets-1d", "fit", "flux-points"])
"""
