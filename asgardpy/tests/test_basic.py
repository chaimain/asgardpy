def test_asgardpyconfig():
    import asgardpy.config as config

    assert hasattr(config, "AsgardpyConfig")


def test_asgardpyanalysis():
    import asgardpy.analysis as ana

    assert hasattr(ana, "AsgardpyAnalysis")
