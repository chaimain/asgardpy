import pytest

from asgardpy.config import AsgardpyConfig


def test_input_dl3(caplog):
    """Testing basic fail safes for DL3Files class."""

    from asgardpy.io.input_dl3 import DL3Files

    config = AsgardpyConfig()
    dl3_dict = config.dataset3d.instruments[0].input_dl3[0]

    dl3_files = DL3Files(dir_dict=dl3_dict, log=None)

    assert dl3_files.log.name == "asgardpy.io.input_dl3"
    assert dl3_files.log.level == 20
    assert caplog.record_tuples[0][2] == "type is not in the expected range for DL3 files"


def test_io_dl4(caplog):
    """Testing basic fail safes for DL4Files class."""

    from asgardpy.io.io_dl4 import DL4Files

    config = AsgardpyConfig()
    dl4_info = config.dataset1d.instruments[0].dl4_dataset_info
    dl4_info.dl4_dataset.glob_pattern["dl4_files"] = "dl4*"
    obs_config = config.dataset1d.instruments[0].dataset_info.observation

    dl4_files = DL4Files(dl4_dataset_info=dl4_info, log=None)

    d_list, _ = dl4_files.get_dl4_files(obs_config)

    assert dl4_files.log.name == "asgardpy.io.io_dl4"
    assert dl4_files.log.level == 20
    assert caplog.record_tuples[0][2] == "No datasets found in ."
    with pytest.raises(ZeroDivisionError):
        1 / len(d_list)


def test_io_dl4_w_models(gammapy_data_path):
    """Testing some more IO options when DL4 have associated models."""

    from asgardpy.io.io_dl4 import DL4Files

    config = AsgardpyConfig()
    dl4_info = config.dataset1d.instruments[0].dl4_dataset_info
    obs_config = config.dataset1d.instruments[0].dataset_info.observation

    dl4_info.dl4_dataset.input_dir = f"{gammapy_data_path}fermi-3fhl-crab/"
    dl4_info.dl4_dataset.glob_pattern["dl4_files"] = "Fermi*data*fits"
    dl4_info.dl4_dataset.glob_pattern["dl4_model_files"] = "Fermi*models.yaml"

    dl4_files = DL4Files(dl4_dataset_info=dl4_info, log=None)
    d_list = dl4_files.get_dl4_dataset(obs_config)

    assert len(d_list) == 1

    dl4_info.dl4_dataset.glob_pattern["dl4_files"] = "Fermi*data*yaml"
    dl4_files = DL4Files(dl4_dataset_info=dl4_info, log=None)
    d_list = dl4_files.get_dl4_dataset(obs_config)

    assert d_list[0].tag == "MapDataset"

    assert dl4_files.read_dl4_file("random.txt") is None
