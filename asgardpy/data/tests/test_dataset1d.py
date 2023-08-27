import pytest


@pytest.mark.test_data
def test_dataset1d_full(base_config_1d):
    from asgardpy.analysis import AsgardpyAnalysis

    analysis = AsgardpyAnalysis(base_config_1d)

    analysis.get_1d_datasets
