import pytest


def test_models_from_config():
    """Test reading models from asgardpy config."""

    from asgardpy.config.generator import AsgardpyConfig, get_model_template
    from asgardpy.data.target import read_models_from_asgardpy_config

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


@pytest.mark.test_data
def test_set_models(base_config, gammapy_data_path):
    """Test non-standard components of Target module."""

    from asgardpy.analysis import AsgardpyAnalysis
    from asgardpy.base.base import PathType
    from asgardpy.data.target import set_models

    analysis = AsgardpyAnalysis(base_config)
    analysis.run(["datasets-3d"])

    analysis.config.target.source_name = "Crab Nebula"
    analysis.config.target.components[
        0
    ].spectral.ebl_abs.filename = f"{gammapy_data_path}ebl/ebl_franceschini_2017.fits.gz"

    data_0, model_0 = set_models(
        analysis.config.target,
        analysis.datasets,
        datasets_name_list=None,
        models=PathType(f"{gammapy_data_path}fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml"),
    )

    with pytest.raises(TypeError):
        _, _ = set_models(
            analysis.config.target,
            analysis.datasets,
            datasets_name_list=None,
            models=f"{gammapy_data_path}fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml",
        )
    assert model_0[0].datasets_names == ["Fermi-LAT_00", "Fermi-LAT_01"]
