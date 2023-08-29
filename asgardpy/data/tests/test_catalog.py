import pytest

from asgardpy.analysis import AsgardpyAnalysis


@pytest.mark.test_data
def test_3d_hess_1d_magic_catalog(gpy_hess_magic):
    analysis = AsgardpyAnalysis(gpy_hess_magic)

    analysis.config.target.use_catalog.selection_radius = "2 deg"

    analysis.run(["datasets-3d"])
    analysis.run(["datasets-1d"])
    analysis.run(["fit"])
    analysis.run(["flux-points"])