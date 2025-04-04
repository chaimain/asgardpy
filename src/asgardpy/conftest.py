import os

import pytest
from gammapy.utils.check import check_tutorials_setup


# add a marker for the tests that need private data and don't run them
# by default
def pytest_configure(config):
    if "test_data" not in config.option.markexpr:  # pragma: no cover
        if config.option.markexpr:
            config.option.markexpr += " and "
        config.option.markexpr += "not test_data"


# Check if gammapy-data is downloaded and if not, then download it.
check_tutorials_setup(download_datasets_path="./gammapy-data")


@pytest.fixture  # (scope="session")
def base_config_path():
    """Get the base config path for basic tests."""

    return "src/asgardpy/tests/config_test_base.yaml"


@pytest.fixture  # (scope="session")
def mwl_config_path():
    """Get the Gammapy MWL tutorial config path."""

    return "src/asgardpy/tests/config_gpy_mwl.yaml"


@pytest.fixture  # (scope="session")
def hess_magic_config_path():
    """Get the config path for HESS (3D) + MAGIC (1D)."""

    return "src/asgardpy/tests/config_test_gadf.yaml"


@pytest.fixture  # (scope="session")
def hawc_config_path():
    """Get the config path for HAWC (3D)."""

    return "src/asgardpy/tests/config_hawc.yaml"


@pytest.fixture  # (scope="session")
def ebl_deabs_path():
    """Get the base config path for basic tests."""

    return "src/asgardpy/tests/config_test_ebl.yaml"


@pytest.fixture  # (scope="session")
def gammapy_data_path():
    """Save a copy of path of gammapy-data for easy and general use."""

    # Check first for the path used in CI test
    if os.path.exists("./gammapy-datasets/1.3/"):
        GAMMAPY_DATA = "./gammapy-datasets/1.3/"
        # Update the environ for builtin EBL models
        os.environ["GAMMAPY_DATA"] = GAMMAPY_DATA
    else:
        # Using the saved path in the environ for users
        GAMMAPY_DATA = os.environ.get("GAMMAPY_DATA", "not set")

    return GAMMAPY_DATA


@pytest.fixture  # (scope="session")
def base_config(base_config_path, gammapy_data_path):
    """Define the base config for basic tests."""

    from asgardpy.config import AsgardpyConfig

    config = AsgardpyConfig().read(base_config_path)

    # Update DL3 file paths
    config.dataset3d.instruments[0].input_dl3[0].input_dir = f"{gammapy_data_path}fermipy-crab/"

    config.dataset3d.instruments[0].input_dl3[1].input_dir = f"{gammapy_data_path}fermipy-crab/"

    config.dataset1d.instruments[0].input_dl3[0].input_dir = f"{gammapy_data_path}hess-dl3-dr1/"

    config.dataset1d.instruments[1].input_dl3[0].input_dir = f"{gammapy_data_path}magic/rad_max/data/"

    return config


@pytest.fixture  # (scope="session")
def base_config_1d(base_config):
    """Define base config for only 1D analysis."""

    base_config_1d = base_config
    base_config_1d.target.source_name = "Crab Nebula"

    # Update model parameters
    base_config_1d.target.components[0].spectral.parameters[0].value = 1.0e-9
    base_config_1d.target.components[0].spectral.parameters[1].value = 0.4
    base_config_1d.target.components[0].spectral.parameters[2].value = 2.0

    base_config_1d.fit_params.fit_range.min = "100 GeV"

    return base_config_1d


@pytest.fixture  # (scope="session")
def gpy_mwl_config(mwl_config_path, gammapy_data_path):
    """Define the Gammapy MWL Tutorial config."""

    from asgardpy.config import AsgardpyConfig, gammapy_model_to_asgardpy_model_config

    config = AsgardpyConfig().read(mwl_config_path)

    # Update DL4 file paths and models file path
    config.target.models_file = f"{gammapy_data_path}fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml"
    config.dataset3d.instruments[
        0
    ].dl4_dataset_info.dl4_dataset.input_dir = f"{gammapy_data_path}fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml"
    config.dataset1d.instruments[
        0
    ].dl4_dataset_info.dl4_dataset.input_dir = f"{gammapy_data_path}joint-crab/spectra/hess/"

    other_config = gammapy_model_to_asgardpy_model_config(config.target.models_file, recursive_merge=False)

    config = config.update(other_config)

    return config


@pytest.fixture  # (scope="session")
def gpy_hess_magic(hess_magic_config_path, gammapy_data_path):
    """Define the config for HESS (3D) + MAGIC (1D)."""

    from asgardpy.config import AsgardpyConfig

    config = AsgardpyConfig().read(hess_magic_config_path)

    # Update DL3 file paths
    config.dataset3d.instruments[0].input_dl3[0].input_dir = f"{gammapy_data_path}hess-dl3-dr1/"

    config.dataset3d.instruments[
        0
    ].dataset_info.background.exclusion.exclusion_file = (
        f"{gammapy_data_path}joint-crab/exclusion/exclusion_mask_crab.fits.gz"
    )

    config.dataset1d.instruments[0].input_dl3[0].input_dir = f"{gammapy_data_path}magic/rad_max/data/"

    return config


@pytest.fixture  # (scope="session")
def ebl_hess_pks(ebl_deabs_path, gammapy_data_path):
    """Define the config for HESS PKS 2155-304 data."""

    from asgardpy.config import AsgardpyConfig

    config = AsgardpyConfig().read(ebl_deabs_path)

    # Update DL4 file path
    config.dataset1d.instruments[0].dl4_dataset_info.dl4_dataset.input_dir = f"{gammapy_data_path}PKS2155-steady/"

    return config


@pytest.fixture  # (scope="session")
def hawc_dl3_config(hawc_config_path, gammapy_data_path):
    """Define the config for HAWC (3D)."""

    from asgardpy.config import AsgardpyConfig

    config = AsgardpyConfig().read(hawc_config_path)

    # Update DL3 file path
    config.dataset3d.instruments[0].input_dl3[0].input_dir = f"{gammapy_data_path}hawc/crab_events_pass4/"

    return config
