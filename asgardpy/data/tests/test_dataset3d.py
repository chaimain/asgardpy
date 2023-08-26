import pytest
import os

@pytest.mark.test_data
def test_dataset3d_full(base_config_path):
    from asgardpy.config import AsgardpyConfig
    from asgardpy.analysis import AsgardpyAnalysis

    config = AsgardpyConfig.read(base_config_path)
    GAMMAPY_DATA = os.environ["GAMMAPY_DATA"]

    # Update DL3 file paths
    config.dataset3d.instruments[0].input_dl3[0].input_dir = f"{GAMMAPY_DATA}fermipy-crab/"

    config.dataset3d.instruments[0].input_dl3[1].input_dir = f"{GAMMAPY_DATA}fermipy-crab/"

    config.dataset1d.instruments[0].input_dl3[0].input_dir = f"{GAMMAPY_DATA}hess-dl3-dr1/"

    analysis = AsgardpyAnalysis(config)

    analysis.run(["datasets-3d"])

    # assert
