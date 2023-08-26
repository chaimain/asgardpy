from gammapy.utils.check import check_tutorials_setup


def test_asgardpyconfig():
    from asgardpy.config import AsgardpyConfig

    assert AsgardpyConfig()


def test_asgardpyanalysis():
    from asgardpy.analysis import AsgardpyAnalysis

    assert AsgardpyAnalysis()


# Check if gammapy-data is downloaded and if not, then download it.
check_tutorials_setup(download_datasets_path="./gammapy-data")
