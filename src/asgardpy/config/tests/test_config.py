import pytest

from asgardpy.config import AsgardpyConfig


def test_config_basic(capsys):
    """Test on basic Config features."""

    from IPython.display import display

    config = AsgardpyConfig()
    assert "AsgardpyConfig\n\n" in str(config)
    assert "AsgardpyConfig(general" in repr(config)
    display(config, display_id="test0")
    captured = capsys.readouterr()
    assert captured.out[:20] == "AsgardpyConfig\n\n    "

    config_str_0 = """
    general:
      n_jobs: 100
    """
    config_100 = AsgardpyConfig.from_yaml(config_str_0)
    assert config_100.general.n_jobs == 100

    with pytest.raises(ValueError):
        config_str_1 = """
        fit_params:
          fit_range:
            min: 10 s
            max: 10 TeV
        """
        AsgardpyConfig.from_yaml(config_str_1)

    with pytest.raises(ValueError):
        config_str_2 = """
        general:
          outdir: ./bla/
        """
        AsgardpyConfig.from_yaml(config_str_2)

    with pytest.raises(ValueError):
        config_str_3 = """
        target:
          sky_position:
            lon: 30 TeV
        """
        AsgardpyConfig.from_yaml(config_str_3)


def test_config_time():
    """Test for reading Time inputs."""
    from pydantic import ValidationError

    config = AsgardpyConfig()

    with pytest.raises(ValidationError):
        config.dataset1d.instruments[0].dataset_info.observation.obs_time = [
            {"format": "abc", "start": "2000-01-01", "stop": "2001-01-01"}
        ]
    with pytest.raises(ValueError):
        config.dataset1d.instruments[0].dataset_info.observation.obs_time = [
            {"format": "iso", "start": "60000", "stop": "2001-01-01"}
        ]
    with pytest.raises(ValueError):
        config.dataset1d.instruments[0].dataset_info.observation.obs_time = [
            {"format": "iso", "start": "2001-01-01", "stop": "60000"}
        ]


def test_get_model_template():
    """Test for reading a model template by a given tag."""

    from asgardpy.config.operations import get_model_template

    new_model = get_model_template("eclp")

    new_config = AsgardpyConfig.read(new_model)

    assert new_config.target.components[0].spectral.type == "ExpCutoffLogParabolaSpectralModel"

    with pytest.raises(IOError):
        new_config.write(new_model, overwrite=False)


def test_create_config_from_dict():
    """Test to create AsgardpyConfig from a simple dict."""

    gen_dict = {"general": {"log": {"level": "warning"}}}
    config = AsgardpyConfig(**gen_dict)

    assert config.general.log.level == "warning"


def test_config_update_gammapy(gammapy_data_path, base_config_1d):
    """Tests to update target model config from Gammapy-based YAML files."""

    import os

    from asgardpy.config.generator import gammapy_model_to_asgardpy_model_config

    main_config = AsgardpyConfig()

    other_config_path = f"{gammapy_data_path}fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml"
    other_config_path_2 = f"{gammapy_data_path}estimators/pks2155_hess_lc/models.yaml"

    other_config_1 = gammapy_model_to_asgardpy_model_config(
        other_config_path,
        recursive_merge=False,
    )
    other_config_2 = gammapy_model_to_asgardpy_model_config(
        other_config_path,
        base_config_1d,
        recursive_merge=False,
    )

    main_config.write("test_base_config.yaml", overwrite=True)
    other_config_3 = gammapy_model_to_asgardpy_model_config(
        other_config_path_2,
        "test_base_config.yaml",
        recursive_merge=False,
    )

    main_config_1 = main_config.update(other_config_1)
    main_config_2 = base_config_1d.update(other_config_2)
    main_config_3 = main_config.update(other_config_3)

    new_spectral_model_name = main_config_1.target.components[0].spectral.type
    new_spectral_model_name_2 = main_config_2.target.components[0].spectral.type
    new_spectral_model_name_3 = main_config_3.target.components[0].spectral.type

    index_max = main_config_3.target.components[0].spectral.parameters[0].max

    assert new_spectral_model_name == "LogParabolaSpectralModel"
    assert new_spectral_model_name_2 == "LogParabolaSpectralModel"
    assert new_spectral_model_name_3 == "PowerLawSpectralModel"
    assert index_max == 10.0

    os.remove("test_base_config.yaml")


def test_config_update():
    """Tests to update target model config from other AsgardpyConfig file."""

    from asgardpy.config.operations import get_model_template

    main_config = AsgardpyConfig()

    spec_model_template_file_1 = get_model_template("bpl")
    spec_model_template_file_2 = get_model_template("sbpl")

    other_config_1 = AsgardpyConfig.read(spec_model_template_file_1)
    other_config_2 = AsgardpyConfig.read(spec_model_template_file_2)

    main_config = main_config.update(other_config_1)
    new_spectral_model_name_1 = main_config.target.components[0].spectral.type

    new_config_str = """
    general:
      n_jobs: 100
    """
    main_config_2 = main_config.update(new_config_str)

    main_config = main_config.update(other_config_2, merge_recursive=True)

    new_spectral_model_name_2 = main_config.target.components[0].spectral.type
    spectral_model_params = main_config.target.components[0].spectral.parameters

    assert new_spectral_model_name_1 == "BrokenPowerLawSpectralModel"
    assert main_config_2.general.n_jobs == 100
    with pytest.raises(TypeError):
        main_config.update(5)
    assert new_spectral_model_name_2 == "SmoothBrokenPowerLawSpectralModel"
    assert len(spectral_model_params) == 6


def test_write_model_config():
    """From a Gammapy Models object, write it as an AsgardpyConfig file."""

    from gammapy.modeling.models import (
        ExpCutoffPowerLaw3FGLSpectralModel,
        Models,
        SkyModel,
    )

    from asgardpy.analysis import AsgardpyAnalysis
    from asgardpy.config.generator import AsgardpyConfig, write_asgardpy_model_to_file
    from asgardpy.config.operations import CONFIG_PATH, get_model_template

    config_ = AsgardpyConfig()
    analysis_ = AsgardpyAnalysis(config_)
    model_ = SkyModel(name="Template", spectral_model=ExpCutoffPowerLaw3FGLSpectralModel())
    model_.spectral_model.index.value = 1.5

    analysis_.final_model = Models(model_)

    assert get_model_template("ecpl-3fgl")

    write_asgardpy_model_to_file(
        gammapy_model=model_,
        output_file=str(CONFIG_PATH) + "/model_templates/model_template_ecpl-3fgl.yaml",
    )

    with pytest.raises(TypeError):
        write_asgardpy_model_to_file()

    write_asgardpy_model_to_file(
        gammapy_model=model_,
        output_file=None,
    )
