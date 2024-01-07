"""
Functions for having interoperatibility of Models type objects with Gammapy
objects.
"""
from astropy.coordinates import SkyCoord
from gammapy.maps import Map
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.models import SPATIAL_MODEL_REGISTRY, SPECTRAL_MODEL_REGISTRY

__all__ = [
    "get_gammapy_spectral_model",
    "params_renaming_to_gammapy",
    "params_rescale_to_gammapy",
    "xml_spatial_model_to_gammapy",
    "xml_spectral_model_to_gammapy",
]


def get_gammapy_spectral_model(spectrum_type, ebl_atten=False, base_model_type="Fermi-XML"):
    """
    Getting the correct Gammapy SpectralModel object after reading a name from a
    different base_model_type.

    Parameter
    ---------
    spectrum_type: str
        Spectral Model type as written in the base model.
    ebl_atten: bool
        If EBL attenuated spectral model needs different treatment.
    base_model_type: str
        Name indicating the XML format used to read the skymodels from.

    Return
    ------
    spectral_model: `gammapy.modleing.models.SpectralModel`
        Gammapy SpectralModel object corresponding to the provided name.
    ebl_atten: bool
        Boolean for EBL attenuated spectral model.
    """
    if base_model_type == "Fermi-XML":
        spectrum_type_no_ebl = spectrum_type.split("EblAtten::")[-1]

        if spectrum_type_no_ebl in ["PLSuperExpCutoff", "PLSuperExpCutoff2"]:
            spectrum_type_final = "ExpCutoffPowerLawSpectralModel"
        elif spectrum_type_no_ebl == "PLSuperExpCutoff4":
            spectrum_type_final = "SuperExpCutoffPowerLaw4FGLDR3SpectralModel"
        else:
            spectrum_type_final = f"{spectrum_type_no_ebl}SpectralModel"

        spectral_model = SPECTRAL_MODEL_REGISTRY.get_cls(spectrum_type_final)()

        if spectrum_type_no_ebl == "LogParabola":
            if "EblAtten" in spectrum_type:
                spectral_model = SPECTRAL_MODEL_REGISTRY.get_cls("PowerLawSpectralModel")()
                ebl_atten = True
            else:
                spectral_model = SPECTRAL_MODEL_REGISTRY.get_cls("LogParabolaSpectralModel")()

    return spectral_model, ebl_atten


def params_renaming_to_gammapy(params_1_name, params_gpy, spectrum_type, params_1_base_model="Fermi-XML"):
    """
    Reading from a given parameter name, get basic parameters details like name,
    unit and is_norm as per Gammapy definition.

    This function collects all such switch cases, based on the base model of the
    given set of parameters.
    """
    if params_1_base_model == "Fermi-XML":
        if params_1_name in ["norm", "prefactor", "integral"]:
            params_gpy["name"] = "amplitude"
            params_gpy["unit"] = "cm-2 s-1 TeV-1"
            params_gpy["is_norm"] = True

        if params_1_name in ["scale", "eb"]:
            params_gpy["name"] = "reference"

        if params_1_name in ["breakvalue"]:
            params_gpy["name"] = "ebreak"

        if params_1_name in ["lowerlimit"]:
            params_gpy["name"] = "emin"

        if params_1_name in ["upperlimit"]:
            params_gpy["name"] = "emax"

        if params_1_name in ["cutoff", "expfactor"]:
            params_gpy["name"] = "lambda_"
            params_gpy["unit"] = "TeV-1"

        if params_1_name in ["index"]:
            params_gpy["name"] = "index"

        if params_1_name in ["index1"]:
            if spectrum_type in ["PLSuperExpCutoff", "PLSuperExpCutoff2"]:
                params_gpy["name"] = "index"
            else:
                params_gpy["name"] = "index1"  # For spectrum_type == "BrokenPowerLaw"

        if params_1_name in ["indexs"]:
            params_gpy["name"] = "index_1"  # For spectrum_type == "PLSuperExpCutoff4"

        if params_1_name in ["index2"]:
            if spectrum_type == "PLSuperExpCutoff4":
                params_gpy["name"] = "index_2"
            elif spectrum_type in ["PLSuperExpCutoff", "PLSuperExpCutoff2"]:
                params_gpy["name"] = "alpha"
            else:
                params_gpy["name"] = "index2"  # For spectrum_type == "BrokenPowerLaw"

        if params_1_name in ["expfactors"]:
            params_gpy["name"] = "expfactor"

    return params_gpy


def params_rescale_to_gammapy(params_gpy, spectrum_type, en_scale_1_to_gpy=1.0e-6, keep_sign=False):
    """
    Rescaling parameters to be used with Gammapy standard units, by using the
    various physical factors (energy for now), compared with the initial set of
    parameters as compared with Gammapy standard units.

    Also, scales the value, min and max of the given parameters, depending on
    their Parameter names.
    """
    if params_gpy["name"] in ["reference", "ebreak", "emin", "emax"]:
        params_gpy["unit"] = "TeV"
        params_gpy["value"] = float(params_gpy["value"]) * float(params_gpy["scale"]) * en_scale_1_to_gpy
        if "error" in params_gpy:
            params_gpy["error"] = float(params_gpy["error"]) * float(params_gpy["scale"]) * en_scale_1_to_gpy
        params_gpy["min"] = float(params_gpy["min"]) * float(params_gpy["scale"]) * en_scale_1_to_gpy
        params_gpy["max"] = float(params_gpy["max"]) * float(params_gpy["scale"]) * en_scale_1_to_gpy
        params_gpy["scale"] = 1.0

    if params_gpy["name"] in ["amplitude"]:
        params_gpy["value"] = float(params_gpy["value"]) * float(params_gpy["scale"]) / en_scale_1_to_gpy
        if "error" in params_gpy:
            params_gpy["error"] = float(params_gpy["error"]) * float(params_gpy["scale"]) / en_scale_1_to_gpy
        params_gpy["min"] = float(params_gpy["min"]) * float(params_gpy["scale"]) / en_scale_1_to_gpy
        params_gpy["max"] = float(params_gpy["max"]) * float(params_gpy["scale"]) / en_scale_1_to_gpy
        params_gpy["scale"] = 1.0

    if params_gpy["name"] in ["index", "index_1", "index_2", "beta"] and not keep_sign:
        # Other than EBL Attenuated Power Law?
        # spectral indices in gammapy are always taken as positive values.
        val_ = float(params_gpy["value"]) * float(params_gpy["scale"])
        if val_ < 0:
            params_gpy["value"] = -1 * val_

            # Reverse the limits while changing the sign
            min_ = -1 * float(params_gpy["min"]) * float(params_gpy["scale"])
            max_ = -1 * float(params_gpy["max"]) * float(params_gpy["scale"])
            params_gpy["min"] = min(min_, max_)
            params_gpy["max"] = max(min_, max_)
            params_gpy["scale"] = 1.0

    if params_gpy["name"] in ["lambda_"] and spectrum_type == "PLSuperExpCutoff":
        # Original parameter is inverse of what gammapy uses
        val_ = float(params_gpy["value"]) * float(params_gpy["scale"])
        params_gpy["value"] = en_scale_1_to_gpy / val_
        if "error" in params_gpy:
            params_gpy["error"] = en_scale_1_to_gpy * float(params_gpy["error"]) / (val_**2)
        min_ = en_scale_1_to_gpy / (float(params_gpy["min"]) * float(params_gpy["scale"]))
        max_ = en_scale_1_to_gpy / (float(params_gpy["max"]) * float(params_gpy["scale"]))
        params_gpy["min"] = min(min_, max_)
        params_gpy["max"] = max(min_, max_)
        params_gpy["scale"] = 1.0

    if float(params_gpy["scale"]) != 1.0:
        # Without any other modifications, but using the scale value
        params_gpy["value"] = float(params_gpy["value"]) * float(params_gpy["scale"])
        if "error" in params_gpy:
            params_gpy["error"] = float(params_gpy["error"]) * float(params_gpy["scale"])
        params_gpy["min"] = float(params_gpy["min"]) * float(params_gpy["scale"])
        params_gpy["max"] = float(params_gpy["max"]) * float(params_gpy["scale"])
        params_gpy["scale"] = 1.0

    return params_gpy


def xml_spectral_model_to_gammapy(
    params, spectrum_type, is_target=False, keep_sign=False, base_model_type="Fermi-XML"
):
    """
    Convert the Spectral Models information from XML model of FermiTools to Gammapy
    standards and return Parameters list.
    Details of the XML model can be seen at
    https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html
    and with examples at
    https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/xml_model_defs.html

    Models from the XML model, not read are -
    ExpCutoff (Blazar modeling with EBL absorption included),
    BPLExpCutoff,
    PLSuperExpCutoff3 (Pulsar modeling),
    BandFunction (GRB analysis)

    Parameters
    ----------
    params: `gammapy.modeling.Parameters`
        List of gammapy Parameter object of a particular Model
    spectrum_type: str
        Spectrum type as defined in XML. To be used only for special cases like
        PLSuperExpCutoff, PLSuperExpCutoff2 and PLSuperExpCutoff4
    is_target: bool
        Boolean to check if the given list of Parameters belong to the target
        source model or not.
    keep_sign: bool
        Boolean to keep the same sign on the parameter values or not.
    base_model_type: str
        Name indicating the XML format used to read the skymodels from.

    Returns
    -------
    params_final: `gammapy.modeling.Parameters`
        Final list of gammapy Parameter object
    """
    new_params = []

    for par in params:
        new_par = {}

        for key_ in par.keys():
            # Getting the "@par_name" for each parameter without the "@"
            if key_ != "@free":
                new_par[key_[1:].lower()] = par[key_]
            else:
                if par["@name"].lower() not in ["scale", "eb"]:
                    new_par["frozen"] = (par[key_] == "0") and not is_target
                else:
                    # Never change frozen status of Reference Energy
                    new_par["frozen"] = par[key_] == "0"
            new_par["unit"] = ""
            new_par["is_norm"] = False

            # Using the nomenclature as used in Gammapy
            new_par = params_renaming_to_gammapy(
                par["@name"].lower(), new_par, spectrum_type, params_1_base_model=base_model_type
            )

        # Some modifications on scaling/sign:
        if base_model_type == "Fermi-XML":
            en_scale_1_to_gpy = 1.0e-6

        new_par = params_rescale_to_gammapy(
            new_par, spectrum_type, en_scale_1_to_gpy=en_scale_1_to_gpy, keep_sign=keep_sign
        )

        if base_model_type == "Fermi-XML":
            if new_par["name"] == "alpha" and spectrum_type in [
                "PLSuperExpCutoff",
                "PLSuperExpCutoff2",
            ]:
                new_par["frozen"] = par["@free"] == "0"

        # Read into Gammapy Parameter object
        new_param = Parameter(name=new_par["name"], value=new_par["value"])
        if "error" in new_par:
            new_param.error = new_par["error"]
        new_param.min = new_par["min"]
        new_param.max = new_par["max"]
        new_param.unit = new_par["unit"]
        new_param.frozen = new_par["frozen"]
        new_param._is_norm = new_par["is_norm"]

        new_params.append(new_param)

    params_final2 = Parameters(new_params)

    # Modifications when different parameters are interconnected
    if base_model_type == "Fermi-XML":
        if spectrum_type == "PLSuperExpCutoff2":
            alpha_inv = 1 / params_final2["alpha"].value
            min_sign = 1
            if params_final2["lambda_"].min < 0:
                min_sign = -1

            params_final2["lambda_"].value = params_final2["lambda_"].value ** alpha_inv / en_scale_1_to_gpy
            params_final2["lambda_"].min = min_sign * (
                abs(params_final2["lambda_"].min) ** alpha_inv / en_scale_1_to_gpy
            )
            params_final2["lambda_"].max = params_final2["lambda_"].max ** alpha_inv / en_scale_1_to_gpy

    return params_final2


def xml_spatial_model_to_gammapy(aux_path, xml_spatial_model, base_model_type="Fermi-XML"):
    """
    Read the spatial model component of the XMl model to Gammapy SpatialModel
    object.

    Details of the Fermi-XML model can be seen at
    https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html
    and with examples at
    https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/xml_model_defs.html

    Parameters
    ----------
    aux_path: `Path`
        Path to the template diffuse models
    xml_spatial_model: `dict`
        Spatial Model component of a particular source from the XML file
    base_model_type: str
        Name indicating the XML format used to read the skymodels from.

    Returns
    -------
    spatial_model: `gammapy.modeling.models.SpatialModel`
        Gammapy Spatial Model object
    """
    spatial_pars = xml_spatial_model["parameter"]

    if base_model_type == "Fermi-XML":
        if xml_spatial_model["@type"] == "SkyDirFunction":
            for par_ in spatial_pars:
                if par_["@name"] == "RA":
                    lon_0 = f"{par_['@value']} deg"
                if par_["@name"] == "DEC":
                    lat_0 = f"{par_['@value']} deg"
            fk5_frame = SkyCoord(
                lon_0,
                lat_0,
                frame="fk5",
            )
            gal_frame = fk5_frame.transform_to("galactic")
            spatial_model = SPATIAL_MODEL_REGISTRY.get_cls("PointSpatialModel")().from_position(gal_frame)

        elif xml_spatial_model["@type"] == "SpatialMap":
            file_name = xml_spatial_model["@file"].split("/")[-1]
            file_path = aux_path / f"Templates/{file_name}"

            spatial_map = Map.read(file_path)
            spatial_map = spatial_map.copy(unit="sr^-1")

            spatial_model = SPATIAL_MODEL_REGISTRY.get_cls("TemplateSpatialModel")(spatial_map, filename=file_path)

        elif xml_spatial_model["@type"] == "RadialGaussian":
            for par_ in spatial_pars:
                if par_["@name"] == "RA":
                    lon_0 = f"{par_['@value']} deg"
                if par_["@name"] == "DEC":
                    lat_0 = f"{par_['@value']} deg"
                if par_["@name"] == "Sigma":
                    sigma = f"{par_['@value']} deg"

            fk5_frame = SkyCoord(
                lon_0,
                lat_0,
                frame="fk5",
            )
            gal_frame = fk5_frame.transform_to("galactic")

            spatial_model = SPATIAL_MODEL_REGISTRY.get_cls("GaussianSpatialModel")(
                lon_0=gal_frame.l, lat_0=gal_frame.b, sigma=sigma, frame="galactic"
            )

    return spatial_model
