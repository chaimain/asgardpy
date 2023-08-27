import pytest

from asgardpy.analysis import AsgardpyAnalysis


@pytest.mark.test_data
def test_dataset3d_full(base_config):
    analysis = AsgardpyAnalysis(base_config)

    analysis.run(["datasets-3d"])

    # assert


@pytest.mark.test_data
def test_dataset3d_different_config(base_config):
    analysis = AsgardpyAnalysis(base_config)

    analysis.config.target.from_3d = True

    analysis.run(["datasets-3d"])
