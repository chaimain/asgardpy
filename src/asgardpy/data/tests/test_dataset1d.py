import pytest

from asgardpy.analysis import AsgardpyAnalysis


@pytest.mark.test_data
def test_dataset1d(base_config_1d):
    """Test for creating 1D stacked DL4 dataset."""

    analysis = AsgardpyAnalysis(base_config_1d)

    analysis.config.dataset1d.instruments[0].dataset_info.safe_mask.methods = ["custom-mask"]
    analysis.config.dataset1d.instruments[0].dataset_info.safe_mask.parameters = {}
    analysis.config.dataset1d.instruments[0].dataset_info.safe_mask.parameters["min"] = "200 GeV"
    analysis.config.dataset1d.instruments[0].dataset_info.safe_mask.parameters["max"] = "20 TeV"

    analysis.get_1d_datasets()


@pytest.mark.test_data
def test_dataset1d_no_stack(base_config_1d):
    """Test for creating 1D unstacked DL4 dataset."""

    analysis = AsgardpyAnalysis(base_config_1d)

    analysis.config.general.stacked_dataset = False
    analysis.config.dataset1d.instruments[0].dataset_info.background.region_finder_method = "reflected"
    analysis.config.dataset1d.instruments[0].dataset_info.background.parameters = {}
    analysis.config.dataset1d.instruments[0].dataset_info.safe_mask.methods = []

    analysis.get_1d_datasets()
