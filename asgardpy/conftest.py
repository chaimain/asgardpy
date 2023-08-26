import pytest


# add a marker for the tests that need private data and don't run them
# by default
def pytest_configure(config):
    if "test_data" not in config.option.markexpr:
        if config.option.markexpr:
            config.option.markexpr += " and "
        config.option.markexpr += "not test_data"


@pytest.mark.test_data
@pytest.fixture  # (scope="session")
def base_config_path():
    """Define the base config for basic tests."""
    return "asgardpy/tests/config_test_base.yaml"
