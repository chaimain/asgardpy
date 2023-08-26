import os

import pytest


@pytest.mark.test_data
def test_dataset3d_full(base_config_path):
    from asgardpy.analysis import AsgardpyAnalysis
    from asgardpy.config import AsgardpyConfig

    config = AsgardpyConfig.read(base_config_path)

    print(os.environ["GAMMAPY_DATA"], "from saved environ path")
    if os.path.exists("./gammapy-datasets/1.1/"):
        GAMMAPY_DATA = "./gammapy-datasets/1.1/"
        os.environ["GAMMAPY_DATA"] = GAMMAPY_DATA
        print(GAMMAPY_DATA, "from existing folder path")
    else:
        GAMMAPY_DATA = os.environ.get("GAMMAPY_DATA", "not set")
        print(GAMMAPY_DATA, "fetched from saved environ path")

    # Update DL3 file paths
    config.dataset3d.instruments[0].input_dl3[0].input_dir = f"{GAMMAPY_DATA}fermipy-crab/"

    config.dataset3d.instruments[0].input_dl3[1].input_dir = f"{GAMMAPY_DATA}fermipy-crab/"

    config.dataset1d.instruments[0].input_dl3[0].input_dir = f"{GAMMAPY_DATA}hess-dl3-dr1/"

    analysis = AsgardpyAnalysis(config)

    analysis.run(["datasets-3d"])

    # assert
