import pytest

from asgardpy.analysis import AsgardpyAnalysis


def test_dataset3d(base_config, caplog):
    """Test for creating 3D DL4 dataset."""

    from gammapy.datasets import MapDataset

    base_config.dataset3d.instruments[0].input_dl3[1].glob_pattern["iso_diffuse"] = ""
    base_config.dataset3d.instruments[0].input_dl3[1].glob_pattern["gal_diffuse"] = ""

    base_config.dataset3d.instruments[0].dataset_info.background.exclusion.regions = []

    analysis = AsgardpyAnalysis(base_config)

    analysis.get_3d_datasets()

    assert len(analysis.datasets) == 2
    assert isinstance(analysis.datasets[0], MapDataset)
    assert analysis.final_model[0].spectral_model.parameters[1].value == 0.015
    assert analysis.datasets[0].counts.geom.npix == (222, 222)
    assert caplog.record_tuples[-3][2] == "Using counts_map to create safe mask"


def test_dataset3d_different_config(base_config, caplog):
    """Test for creating 3D DL4 dataset with target model info from DL3 files."""

    analysis_0 = AsgardpyAnalysis(base_config)

    analysis_0.config.target.from_3d = True
    analysis_0.config.dataset3d.instruments[0].dataset_info.geom.from_events_file = False

    analysis_0.get_3d_datasets()

    assert analysis_0.final_model[0].spectral_model.parameters[1].value == 0.01
    assert analysis_0.datasets[0].counts.geom.npix == (40, 40)

    # Use coordinates for the central region of the exclusion mask
    base_config.dataset3d.instruments[0].dataset_info.background.exclusion.regions[0].name = ""

    analysis_1 = AsgardpyAnalysis(base_config)
    analysis_1.config.dataset3d.instruments[0].dataset_info.key = []

    analysis_1.get_3d_datasets()
    print(caplog.record_tuples[-4][2])
    assert caplog.record_tuples[-4][2][:15] == "No distinct key"

    with pytest.raises(ValueError):
        analysis_2 = AsgardpyAnalysis(base_config)
        analysis_2.config.dataset3d.instruments[0].dataset_info.key = ["12"]
        analysis_2.get_3d_datasets()


def test_fermi_fits_file(gammapy_data_path):
    """Basic test on I/O of Fermi-LAT Fits files."""

    from astropy.io import fits

    from asgardpy.base.geom import get_source_position
    from asgardpy.config import AsgardpyConfig

    config = AsgardpyConfig()
    fits_file = f"{gammapy_data_path}fermipy-crab/ft1_test.fits"
    fits_header = fits.open(fits_file)[1].header

    source_pos = get_source_position(config.target.sky_position, fits_header)

    assert source_pos["center"].ra.deg == 83.633


def test_hawc_analysis(hawc_dl3_config):
    """Basic test on running analysis of HAWC DL3 data."""

    from asgardpy.analysis import AsgardpyAnalysis

    analysis = AsgardpyAnalysis(hawc_dl3_config)

    analysis.run()
    flux_table = analysis.flux_points[0].to_table(sed_type="e2dnde", formatted=True, format="gadf-sed")

    assert flux_table["counts"][3].sum() == 463.0
