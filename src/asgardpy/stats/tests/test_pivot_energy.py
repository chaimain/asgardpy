import pytest

from asgardpy.analysis import AsgardpyAnalysis
from asgardpy.stats import fetch_pivot_energy


@pytest.mark.test_data
def test_get_pivot_energy(gpy_hess_magic):
    """
    Check the pivot energy for given fit model on a dataset.
    """
    analysis = AsgardpyAnalysis(gpy_hess_magic)

    analysis.run(["datasets-3d", "datasets-1d"])

    e_ref = fetch_pivot_energy(analysis)

    assert e_ref.value == 0.20085434771049843


@pytest.mark.test_data
def test_get_pivot_energy_from_start(gpy_hess_magic):
    """
    Check the pivot energy for given fit model on a dataset from the start of
    the AsgardpyAnalysis object. Test using the custom SpectralModel of ECLP
    and without any associated EBL absorption model.
    """
    from asgardpy.config.generator import get_model_template

    new_model = get_model_template("eclp")
    gpy_hess_magic.target.models_file = new_model
    gpy_hess_magic.target.components[0].spectral.ebl_abs.reference = ""

    analysis = AsgardpyAnalysis(gpy_hess_magic)

    e_ref = fetch_pivot_energy(analysis)

    if analysis.fit_result.success:
        assert e_ref.value == 0.030128153004345924
    else:
        # For windows OS CI, the fit fails
        assert e_ref.value == 0.0014998909471849885
