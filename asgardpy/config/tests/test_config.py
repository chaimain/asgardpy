import pytest


@pytest.mark.test_data
def test_config(base_config_path):
    from asgardpy.config import AsgardpyConfig

    con = AsgardpyConfig.read(base_config_path)
