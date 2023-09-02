import pytest

from asgardpy.analysis import AsgardpyAnalysis


@pytest.mark.test_data
def test_joint_3d_1d(base_config):
    """
    Test to run a major Fermi (3D) + HESS (1D) + MAGIC (1D) joint analysis.
    """

    analysis = AsgardpyAnalysis(base_config)

    analysis.run(["datasets-3d"])
    analysis.run(["datasets-1d"])
    analysis.run_fit
