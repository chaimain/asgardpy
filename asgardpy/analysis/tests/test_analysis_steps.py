import pytest

from asgardpy.analysis import AsgardpyAnalysis


@pytest.mark.test_data
def test_only_3d(base_config):
    analysis = AsgardpyAnalysis(base_config)

    analysis.config.fit_params.fit_range.max = "500 GeV"

    analysis.run(["datasets-3d"])
    analysis.run(["fit"])
    # analysis.run(["flux-points"])


@pytest.mark.test_data
def test_only_1d(base_config_1d):
    analysis = AsgardpyAnalysis(base_config_1d)

    analysis.config.fit_params.fit_range.min = "500 GeV"

    analysis.run(["datasets-1d"])
    analysis.run(["fit"])
    analysis.run(["flux-points"])


@pytest.mark.test_data
def test_joint_3d_1d(base_config):
    analysis = AsgardpyAnalysis(base_config)

    analysis.run(["datasets-3d"])
    analysis.run(["datasets-1d"])
    analysis.run(["fit"])
    # analysis.run(["flux-points"])
