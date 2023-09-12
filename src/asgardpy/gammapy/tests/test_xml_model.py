import logging

import pytest
from gammapy.modeling.models import Models

from asgardpy.config import AsgardpyConfig
from asgardpy.data.dataset_3d import Dataset3DGeneration


@pytest.mark.test_data
def test_xml(gammapy_data_path):
    """Test reading various different Fermi-XML source models."""

    config = AsgardpyConfig()
    log = logging.getLogger("Test XML model")

    dataset_3d = config.dataset3d.instruments[0]
    dataset_3d.input_dl3[0].input_dir = f"{gammapy_data_path}fermipy-crab/"
    aux_path = dataset_3d.input_dl3[0].input_dir

    xml_file = aux_path / "test_fermi_xml_models.xml"
    config.target.source_name = "4FGL J0534.5+2201i"
    config.target.from_3d = True

    data_3d = Dataset3DGeneration(log, dataset_3d, config)
    data_3d.get_list_objects(aux_path, xml_file)
    data_3d.list_sources = Models(data_3d.list_sources)

    assert data_3d.list_sources[0].spatial_model.tag[0] == "GaussianSpatialModel"
    assert data_3d.list_sources[0].spectral_model.model1.tag[1] == "lp"
    assert data_3d.list_sources[1].spatial_model.tag[0] == "TemplateSpatialModel"
    assert data_3d.list_sources[2].spatial_model.tag[0] == "PointSpatialModel"
    assert data_3d.list_sources[2].spectral_model.tag[1] == "ecpl"
    assert data_3d.list_sources[3].spectral_model.tag[1] == "pl"
    assert data_3d.list_sources[4].spectral_model.tag[1] == "ecpl"
    assert data_3d.list_sources[5].spectral_model.tag[1] == "secpl-4fgl-dr3"
    assert data_3d.list_sources[6].spectral_model.tag[1] == "bpl"
    assert data_3d.list_sources[7].spectral_model.tag[1] == "sbpl"
    assert data_3d.list_sources[8].spectral_model.tag[1] == "pl"
    assert data_3d.list_sources[8].spatial_model.tag[0] == "GaussianSpatialModel"
