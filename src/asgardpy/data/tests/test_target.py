import pytest


def test_models_from_config():
    """Test reading models from asgardpy config."""

    from asgardpy.config.generator import AsgardpyConfig, get_model_template
    from asgardpy.data.target import read_models_from_asgardpy_config, set_models

    config_eclp = AsgardpyConfig.read(get_model_template("eclp"))
    config_bpl2 = AsgardpyConfig.read(get_model_template("bpl2"))
    config_lp = AsgardpyConfig.read(get_model_template("lp"))
    config_fov = AsgardpyConfig.read(get_model_template("fov"))

    config_eclp.target.components[0].spectral.ebl_abs.reference = ""
    config_bpl2.target.components[0].spectral.ebl_abs.reference = ""
    config_lp.target.components[0].spectral.ebl_abs.reference = ""
    config_fov.target.components[0].spectral.ebl_abs.reference = ""

    model_eclp = read_models_from_asgardpy_config(config_eclp.target)
    model_bpl2 = read_models_from_asgardpy_config(config_bpl2.target)
    model_lp = read_models_from_asgardpy_config(config_lp.target)
    model_fov = read_models_from_asgardpy_config(config_fov.target)

    assert model_eclp[0].spectral_model.tag[0] == "ExpCutoffLogParabolaSpectralModel"
    assert model_bpl2[0].spectral_model.tag[0] == "BrokenPowerLaw2SpectralModel"
    assert model_lp[0].spectral_model.tag[0] == "LogParabolaSpectralModel"
    assert model_fov[0].spectral_model.tag[0] == "PowerLawNormSpectralModel"
    assert model_fov[0].spatial_model.tag[0] == "ConstantSpatialModel"

    # Exception for empty models information in config.
    with pytest.raises(Exception):
        _, _ = set_models()


@pytest.mark.test_data
def test_set_models(base_config, gammapy_data_path):
    """Test non-standard components of Target module."""

    from asgardpy.analysis import AsgardpyAnalysis
    from asgardpy.base.base import PathType
    from asgardpy.data.target import set_models

    analysis_0 = AsgardpyAnalysis(base_config)
    analysis_1 = AsgardpyAnalysis(base_config)

    # Check when using create_source_skymodel function
    analysis_0.run(["datasets-3d"])
    # Check when using read_models_from_asgardpy_config
    analysis_1.run(["datasets-1d"])

    analysis_0.config.target.source_name = "Crab Nebula"
    analysis_1.config.target.source_name = "Crab Nebula"

    ebl_file_name = "ebl_franceschini_2017.fits.gz"
    ebl_file = f"{gammapy_data_path}ebl/{ebl_file_name}"
    model_file_0 = f"{gammapy_data_path}fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml"
    model_file_1 = f"{gammapy_data_path}fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml"

    analysis_0.config.target.components[0].spectral.ebl_abs.filename = ebl_file
    analysis_1.config.target.components[0].spectral.ebl_abs.filename = ebl_file

    data_0, model_0 = set_models(
        analysis_0.config.target,
        analysis_0.datasets,
        datasets_name_list=None,
        models=PathType(model_file_0),
    )

    data_1, model_1 = set_models(
        analysis_0.config.target,
        analysis_0.datasets,
        datasets_name_list=None,
    )
    data_2, model_2 = set_models(
        analysis_1.config.target,
        analysis_1.datasets,
        datasets_name_list=None,
    )

    with pytest.raises(TypeError):
        _, _ = set_models(
            analysis_0.config.target,
            analysis_0.datasets,
            datasets_name_list=None,
            models=model_file_1,
        )
    assert model_0[0].datasets_names == ["Fermi-LAT_00", "Fermi-LAT_01"]
    assert model_1[0].spectral_model.model2.filename.name == ebl_file_name
    assert model_2[0].spectral_model.model2.filename.name == ebl_file_name
