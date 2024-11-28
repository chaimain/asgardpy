from pathlib import Path

from gammapy.modeling.models import Models

from asgardpy.gammapy.read_models import read_fermi_xml_models_list


def test_xml_only_source_models(gammapy_data_path):
    """Test reading various different Fermi-XML source models."""
    from asgardpy.config import AsgardpyConfig

    config = AsgardpyConfig()
    list_source_models = []

    dl3_aux_path = Path(f"{gammapy_data_path}fermipy-crab/")

    diffuse_models = {}

    xml_file = dl3_aux_path / "test_fermi_xml_models.xml"
    config.target.source_name = "4FGL J0534.5+2201i"
    config.target.from_3d = True

    list_source_models, _ = read_fermi_xml_models_list(
        list_source_models, dl3_aux_path, xml_file, diffuse_models, config.target
    )
    list_source_models = Models(list_source_models)

    assert list_source_models[0].spatial_model.tag[0] == "GaussianSpatialModel"
    assert list_source_models[0].spectral_model.tag[1] == "lp"
    assert list_source_models[1].spatial_model.tag[0] == "TemplateSpatialModel"
    assert list_source_models[2].spatial_model.tag[0] == "PointSpatialModel"
    assert list_source_models[2].spectral_model.tag[1] == "ecpl"
    assert list_source_models[3].spectral_model.tag[1] == "pl"
    assert list_source_models[4].spectral_model.tag[1] == "ecpl"
    assert list_source_models[5].spectral_model.tag[1] == "secpl-4fgl-dr3"
    assert list_source_models[6].spectral_model.tag[1] == "bpl"
    assert list_source_models[7].spectral_model.tag[1] == "sbpl"
    assert list_source_models[8].spectral_model.tag[1] == "pl"
    assert list_source_models[8].spatial_model.tag[0] == "GaussianSpatialModel"
    assert list_source_models[-1].spectral_model.tag[1] == "pl-2"


def test_xml_with_diffuse_models(gammapy_data_path):
    """Test reading Fermi-XML models with diffuse models included."""

    list_source_models = []
    diffuse_models = {}

    dl3_aux_path = Path(f"{gammapy_data_path}fermipy-crab/")

    xml_file = dl3_aux_path / "srcmdl_00.xml"
    diffuse_models["gal_diffuse"] = dl3_aux_path / "gll_iem_v07_cutout.fits"
    diffuse_models["iso_diffuse"] = dl3_aux_path / "iso_P8R3_SOURCE_V3_FRONT_v1.txt"
    diffuse_models["key_name"] = None

    list_source_models, diffuse_models = read_fermi_xml_models_list(
        list_source_models, dl3_aux_path, xml_file, diffuse_models
    )
    list_source_models = Models(list_source_models)

    assert list_source_models[1].name == "4FGL J0534.5+2201s"
    assert list_source_models[-1].name == "diffuse-iem"
    assert list_source_models[-2].name == "fermi-diffuse-iso"
