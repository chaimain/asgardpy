from gammapy.utils.check import check_tutorials_setup


def test_asgardpyconfig():
    import asgardpy.config as config

    assert hasattr(config, "AsgardpyConfig")


def test_asgardpyanalysis():
    import asgardpy.analysis as ana

    assert hasattr(ana, "AsgardpyAnalysis")


# Check if gammapy-data is downloaded and if not, then download it.
check_tutorials_setup(download_datasets_path="./gammapy-data")
