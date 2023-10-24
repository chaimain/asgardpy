import pytest

from asgardpy.analysis import AsgardpyAnalysis


@pytest.mark.test_data
def test_3d_hess_1d_magic_catalog(gpy_hess_magic):
    """
    Test the usage of Source Catalog for FoV background source models.
    """

    analysis = AsgardpyAnalysis(gpy_hess_magic)

    analysis.config.target.use_catalog.selection_radius = "10 deg"
    analysis.config.target.roi_selection.roi_radius = "2.8 deg"

    analysis.config.general.steps = ["datasets-3d", "datasets-1d"]

    analysis.run()

    assert "3FHL J0536.2+1733" in analysis.final_model.names
    assert len(list(analysis.final_model.parameters.free_parameters)) == 31
