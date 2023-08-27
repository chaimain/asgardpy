import os

import pytest
from gammapy.utils.check import check_tutorials_setup


# add a marker for the tests that need private data and don't run them
# by default
def pytest_configure(config):
    if "test_data" not in config.option.markexpr:
        if config.option.markexpr:
            config.option.markexpr += " and "
        config.option.markexpr += "not test_data"


# Check if gammapy-data is downloaded and if not, then download it.
check_tutorials_setup(download_datasets_path="./gammapy-data")


@pytest.mark.test_data
@pytest.fixture  # (scope="session")
def base_config_path():
    """Get the base config path for basic tests."""
    return "asgardpy/tests/config_test_base.yaml"


@pytest.mark.test_data
@pytest.fixture  # (scope="session")
def base_config(base_config_path):
    """Define the base config for basic tests."""
    from asgardpy.config import AsgardpyConfig

    config = AsgardpyConfig().read(base_config_path)

    # Check first for the path used in CI test
    if os.path.exists("./gammapy-datasets/1.1/"):
        GAMMAPY_DATA = "./gammapy-datasets/1.1/"
        # Update the environ for builtin EBL models
        os.environ["GAMMAPY_DATA"] = GAMMAPY_DATA
    else:
        # Using the saved path in the environ for users
        GAMMAPY_DATA = os.environ.get("GAMMAPY_DATA", "not set")

    # Update DL3 file paths
    config.dataset3d.instruments[0].input_dl3[0].input_dir = f"{GAMMAPY_DATA}fermipy-crab/"

    config.dataset3d.instruments[0].input_dl3[1].input_dir = f"{GAMMAPY_DATA}fermipy-crab/"

    config.dataset1d.instruments[0].input_dl3[0].input_dir = f"{GAMMAPY_DATA}hess-dl3-dr1/"

    return config


@pytest.mark.test_data
@pytest.fixture  # (scope="session")
def base_config_1d(base_config):
    base_config_1d = base_config
    base_config_1d.target.source_name = "Crab Nebula"

    base_config_1d.target.components[0].spectral.parameters[0].value = 1.0e-8
    base_config_1d.target.components[0].spectral.parameters[1].value = 400
    base_config_1d.target.components[0].spectral.parameters[2].value = 2.5

    base_config_1d.fit_params.fit_range.min = "100 GeV"

    return base_config_1d
