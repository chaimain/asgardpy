from regions import PointSkyRegion

from asgardpy.analysis import AsgardpyAnalysis


def test_gpy_mwl(gpy_mwl_config, gammapy_data_path):
    """
    Test for running the 3D+1D joint analysis tutorial example from Gammapy.
    """

    from gammapy.datasets import FluxPointsDataset
    from gammapy.estimators import FluxPoints
    from gammapy.modeling.models import create_crab_spectral_model

    from asgardpy.data.target import set_models

    analysis = AsgardpyAnalysis(gpy_mwl_config)

    # Update model parameters
    # LP-amplitude
    analysis.config.target.components[0].spectral.parameters[0].value /= 1e4
    analysis.config.target.components[0].spectral.parameters[0].min = 1.0e-13
    analysis.config.target.components[0].spectral.parameters[0].max = 0.01
    analysis.config.target.components[0].spectral.parameters[0].frozen = False

    # LP-reference
    analysis.config.target.components[0].spectral.parameters[1].value *= 1e3
    analysis.config.target.components[0].spectral.parameters[1].min = 0.001
    analysis.config.target.components[0].spectral.parameters[1].max = 100

    # LP-alpha
    analysis.config.target.components[0].spectral.parameters[2].min = 0.5
    analysis.config.target.components[0].spectral.parameters[2].max = 5.0
    analysis.config.target.components[0].spectral.parameters[2].frozen = False

    # LP-beta
    analysis.config.target.components[0].spectral.parameters[3].min = 0.001
    analysis.config.target.components[0].spectral.parameters[3].max = 1.0
    analysis.config.target.components[0].spectral.parameters[3].frozen = False

    # Spatial-lon
    analysis.config.target.components[0].spatial.parameters[0].error = 1.0e-6
    analysis.config.target.components[0].spatial.parameters[0].min = 83.0
    analysis.config.target.components[0].spatial.parameters[0].max = 84.0

    # Spatial-lat
    analysis.config.target.components[0].spatial.parameters[1].error = 1.0e-6
    analysis.config.target.components[0].spatial.parameters[1].min = -90
    analysis.config.target.components[0].spatial.parameters[1].max = +90

    # FoV-bkg-Norm - Not being read exactly
    analysis.config.target.components[1].spectral.parameters[0].min = 0.0
    analysis.config.target.components[1].spectral.parameters[0].max = 10.0
    analysis.config.target.components[1].spectral.parameters[0].frozen = False

    analysis.run(["datasets-3d"])
    analysis.run(["datasets-1d"])

    # Include HAWC Flux Points
    # Read to Gammapy objects
    filename = f"{gammapy_data_path}hawc_crab/HAWC19_flux_points.fits"
    fp_hawc = FluxPoints.read(filename, reference_model=create_crab_spectral_model("meyer"))
    fpd_hawc = FluxPointsDataset(data=fp_hawc, name="HAWC")

    analysis.datasets.append(fpd_hawc)

    # Update other dataset info
    analysis.dataset_name_list.append("HAWC")

    """
    # FPE to only run for Fermi and HESS datasets, as HAWC is already estimated.
    analysis.instrument_spectral_info["name"].append("HAWC")

    hawc_en = np.array([1, 1.78, 3.16, 5.62, 10.0, 17.8, 31.6, 56.2, 100, 177, 316]) * u.TeV
    analysis.instrument_spectral_info["spectral_energy_ranges"].append(hawc_en)
    analysis.instrument_spectral_info["en_bins"] += 10
    analysis.instrument_spectral_info["DoF"] += 10
    """

    # Reset models to the updated dataset
    analysis.datasets, analysis.final_model = set_models(
        analysis.config.target,
        analysis.datasets,
        analysis.dataset_name_list,
        models=analysis.final_model,
    )

    # Update Fit energy range
    analysis.config.fit_params.fit_range.max = "300 TeV"

    analysis.run(["fit"])
    analysis.get_flux_points()

    assert analysis.fit_result.success is True
    assert len(analysis.datasets) == 3
    assert len(analysis.flux_points) == 2
    assert analysis.datasets[1].counts.geom.region is None


def test_3d_hess_1d_magic(gpy_hess_magic):
    """Test for running HESS (3D) + MAGIC (1D) joint analysis."""

    analysis = AsgardpyAnalysis(gpy_hess_magic)

    analysis.run(["datasets-3d", "datasets-1d", "fit"])

    assert int(analysis.datasets[0].gti.time_sum.value) == 5056
    assert isinstance(analysis.datasets[1].counts.geom.region, PointSkyRegion)
