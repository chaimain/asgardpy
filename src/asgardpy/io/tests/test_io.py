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
    obs_config = config.dataset1d.instruments[0].dataset_info.observation

    dl4_files = DL4Files(dl4_dataset_info=dl4_info, log=None)

    d_list = dl4_files.get_dl4_files(obs_config)

    assert dl4_files.log.name == "asgardpy.io.io_dl4"
    assert dl4_files.log.level == 20
    assert caplog.record_tuples[0][2] == "No datasets found in ."
    with pytest.raises(ZeroDivisionError):
        1 / len(d_list)
