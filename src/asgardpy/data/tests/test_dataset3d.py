import pytest

from asgardpy.analysis import AsgardpyAnalysis


@pytest.mark.test_data
def test_dataset3d(base_config):
    """Test for creating 3D DL4 dataset."""

    from gammapy.datasets import MapDataset

    analysis = AsgardpyAnalysis(base_config)

    analysis.get_3d_datasets()

    assert len(analysis.datasets) == 2
    assert isinstance(analysis.datasets[0], MapDataset)
    assert analysis.final_model[0].spectral_model.parameters[1].value == 0.015
    assert analysis.datasets[0].counts.geom.npix == (222, 222)


@pytest.mark.test_data
def test_dataset3d_different_config(base_config):
    """Test for creating 3D DL4 dataset with target model info from DL3 files."""

    analysis = AsgardpyAnalysis(base_config)

    analysis.config.target.from_3d = True
    analysis.config.dataset3d.instruments[0].dataset_info.geom.from_events_file = False

    analysis.get_3d_datasets()

    assert analysis.final_model[0].spectral_model.parameters[1].value == 0.01
    assert analysis.datasets[0].counts.geom.npix == (40, 40)
