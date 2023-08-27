import pytest

from asgardpy.analysis import AsgardpyAnalysis


@pytest.mark.test_data
def test_only_3d(base_config):
    analysis = AsgardpyAnalysis(base_config)

    analysis.config.fit_params.fit_range.max = "500 GeV"

    analysis.run(["datasets-3d", "fit", "flux-points"])


@pytest.mark.test_data
def test_only_1d(base_config_1d):
    analysis = AsgardpyAnalysis(base_config_1d)

    # amplitude
    analysis.config.target.components[0].spectral.parameters[0].value = 1e-9
    analysis.config.target.components[0].spectral.parameters[0].error = 1e-10

    # reference
    analysis.config.target.components[0].spectral.parameters[1].value = 0.4

    # alpha
    analysis.config.target.components[0].spectral.parameters[2].value = 2.0

    analysis.config.fit_params.fit_range.min = "500 GeV"

    analysis.run(["datasets-1d", "fit", "flux-points"])


@pytest.mark.test_data
def test_joint_3d_1d(base_config):
    analysis = AsgardpyAnalysis(base_config)

    analysis.run()
