import pytest


@pytest.mark.test_data
def test_config_update_gammapy(gammapy_data_path):
    from asgardpy.config import AsgardpyConfig, gammapy_to_asgardpy_model_config

    main_config = AsgardpyConfig()

    other_config_path = f"{gammapy_data_path}fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml"
    other_config = gammapy_to_asgardpy_model_config(other_config_path)

    main_config = main_config.update(other_config)

    new_spectral_model_name = main_config.target.components[0].spectral.type

    assert new_spectral_model_name == "LogParabolaSpectralModel"


def test_config_update():
    from asgardpy.config.generator import CONFIG_PATH, AsgardpyConfig

    main_config = AsgardpyConfig()

    spec_model_template_files = sorted(
        list(CONFIG_PATH.glob("model_templates/model_template*yaml"))
    )

    other_config = AsgardpyConfig.read(spec_model_template_files[0])

    main_config = main_config.update(other_config)

    new_spectral_model_name = main_config.target.components[0].spectral.type

    assert new_spectral_model_name == "BrokenPowerLawSpectralModel"
