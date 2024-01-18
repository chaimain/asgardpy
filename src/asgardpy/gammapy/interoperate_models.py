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

        match spectrum_type_no_ebl:
            case "PLSuperExpCutoff" | "PLSuperExpCutoff2":
                spectrum_type_final = "ExpCutoffPowerLawSpectralModel"
            case "PLSuperExpCutoff4":
                spectrum_type_final = "SuperExpCutoffPowerLaw4FGLDR3SpectralModel"
            case _:
                spectrum_type_final = f"{spectrum_type_no_ebl}SpectralModel"

        spectral_model = SPECTRAL_MODEL_REGISTRY.get_cls(spectrum_type_final)()

        if spectrum_type_no_ebl == "LogParabola":
            if "EblAtten" in spectrum_type:
                spectral_model = SPECTRAL_MODEL_REGISTRY.get_cls("PowerLawSpectralModel")()
                ebl_atten = True
            else:
                spectral_model = SPECTRAL_MODEL_REGISTRY.get_cls("LogParabolaSpectralModel")()

    return spectral_model, ebl_atten


def rename_fermi_energy_params(param_name):
    """
    Simple match-case for renaming Energy parameters from Fermi-XML model to
    Gammapy standard.
    """
    match param_name:
        case "scale" | "eb":
            return "reference"

        case "breakvalue":
            return "ebreak"

        case "lowerlimit":
            return "emin"

        case "upperlimit":
            return "emax"


def rename_fermi_index_params(param_name, spectrum_type):
    """
    Simple match-case for renaming spectral index parameters from Fermi-XML
    model to Gammapy standard.

    Some spectral models share common names in Fermi-XML but not in Gammapy.
    """
    match param_name:
        case "index":
            return "index"

        case "index1":
            match spectrum_type:
                case "PLSuperExpCutoff" | "PLSuperExpCutoff2":
                    return "index"
                case _:
                    return "index1"  # For spectrum_type == "BrokenPowerLaw"

        case "indexs":
            return "index_1"  # For spectrum_type == "PLSuperExpCutoff4"

        case "index2":
            match spectrum_type:
                case "PLSuperExpCutoff4":
                    return "index_2"
                case "PLSuperExpCutoff" | "PLSuperExpCutoff2":
                    return "alpha"
                case _:
                    return "index2"  # For spectrum_type == "BrokenPowerLaw"


def params_renaming_to_gammapy(params_1_name, params_gpy, spectrum_type, params_1_base_model="Fermi-XML"):
    """
    Reading from a given parameter name, get basic parameters details like name,
    unit and is_norm as per Gammapy definition.

    This function collects all such switch cases, based on the base model of the
    given set of parameters.
    """
    if params_1_base_model == "Fermi-XML":
        match params_1_name:
            case "norm" | "prefactor" | "integral":
                params_gpy["name"] = "amplitude"
                params_gpy["is_norm"] = True

                match spectrum_type:
                    case "PowerLaw2":
                        params_gpy["unit"] = "cm-2 s-1"
                    case _:
                        params_gpy["unit"] = "cm-2 s-1 TeV-1"

            case "scale" | "eb" | "breakvalue" | "lowerlimit" | "upperlimit":
                params_gpy["name"] = rename_fermi_energy_params(params_1_name)

            case "index" | "index1" | "indexs" | "index2":
                params_gpy["name"] = rename_fermi_index_params(params_1_name, spectrum_type)

            case "cutoff" | "expfactor":
                params_gpy["name"] = "lambda_"
                params_gpy["unit"] = "TeV-1"

            case "expfactors":
                params_gpy["name"] = "expfactor"

    return params_gpy


def rescale_parameters(params_dict, scale_factor):
    """
    Function to rescale the value of the parameters as per Gammapy standard,
    using a multiplying factor, considering the final scale value to be 1.
    """
    params_dict["value"] = float(params_dict["value"]) * float(params_dict["scale"]) * scale_factor

    if "error" in params_dict:
        params_dict["error"] = float(params_dict["error"]) * float(params_dict["scale"]) * scale_factor

    if scale_factor == -1:
        # Reverse the limits while changing the sign
        min_ = float(params_dict["min"]) * float(params_dict["scale"]) * scale_factor
        max_ = float(params_dict["max"]) * float(params_dict["scale"]) * scale_factor

        params_dict["min"] = min(min_, max_)
        params_dict["max"] = max(min_, max_)
    else:
        params_dict["min"] = float(params_dict["min"]) * float(params_dict["scale"]) * scale_factor
        params_dict["max"] = float(params_dict["max"]) * float(params_dict["scale"]) * scale_factor

    params_dict["scale"] = 1.0

    return params_dict


def invert_parameters(params_dict, scale_factor):
    """
    Function to rescale the value of the parameters as per Gammapy standard,
    when the main parameter value needs to be inverted, considering the final
    scale value to be 1.
    """
    val_ = float(params_dict["value"]) * float(params_dict["scale"])

    params_dict["value"] = scale_factor / val_

    if "error" in params_dict:
        params_dict["error"] = scale_factor * float(params_dict["error"]) / (val_**2)

    min_ = scale_factor / (float(params_dict["min"]) * float(params_dict["scale"]))
    max_ = scale_factor / (float(params_dict["max"]) * float(params_dict["scale"]))

    params_dict["min"] = min(min_, max_)
    params_dict["max"] = max(min_, max_)

    params_dict["scale"] = 1.0

    return params_dict


def params_rescale_to_gammapy(params_gpy, spectrum_type, en_scale_1_to_gpy=1.0e-6, keep_sign=False):
    """
    Rescaling parameters to be used with Gammapy standard units, by using the
    various physical factors (energy for now), compared with the initial set of
    parameters as compared with Gammapy standard units.

    Also, scales the value, min and max of the given parameters, depending on
    their Parameter names.
    """
    match params_gpy["name"]:
        case "reference" | "ebreak" | "emin" | "emax":
            params_gpy["unit"] = "TeV"
            params_gpy = rescale_parameters(params_gpy, en_scale_1_to_gpy)

        case "amplitude":
            params_gpy = rescale_parameters(params_gpy, 1 / en_scale_1_to_gpy)

        case "index" | "index_1" | "index_2" | "beta":
            if not keep_sign:
                # Other than EBL Attenuated Power Law?
                # spectral indices in gammapy are always taken as positive values.
                val_ = float(params_gpy["value"]) * float(params_gpy["scale"])
                if val_ < 0:
                    params_gpy = rescale_parameters(params_gpy, -1)

        case "lambda_":
            if spectrum_type == "PLSuperExpCutoff":
                # Original parameter is inverse of what gammapy uses
                params_gpy = invert_parameters(params_gpy, en_scale_1_to_gpy)

    if float(params_gpy["scale"]) != 1.0:
        params_gpy = rescale_parameters(params_gpy, 1)

    return params_gpy


def param_dict_to_gammapy_parameter(new_par):
    """Read a dict object into Gammapy Parameter object."""
    new_param = Parameter(name=new_par["name"], value=new_par["value"])

    if "error" in new_par:
        new_param.error = new_par["error"]

    new_param.min = new_par["min"]
    new_param.max = new_par["max"]
    new_param.unit = new_par["unit"]
    new_param.frozen = new_par["frozen"]
    new_param._is_norm = new_par["is_norm"]

    return new_param


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

    # Some modifications on scaling/sign
    # By default for base_model_type == "Fermi-XML"
    en_scale_1_to_gpy = 1.0e-6

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

        new_par = params_rescale_to_gammapy(
            new_par, spectrum_type, en_scale_1_to_gpy=en_scale_1_to_gpy, keep_sign=keep_sign
        )

        if base_model_type == "Fermi-XML":
            if new_par["name"] == "alpha" and spectrum_type in [
                "PLSuperExpCutoff",
                "PLSuperExpCutoff2",
            ]:
                new_par["frozen"] = par["@free"] == "0"

        new_params.append(param_dict_to_gammapy_parameter(new_par))

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


def fetch_spatial_galactic_frame(spatial_params, sigma=False):
    """
    Function to get the galactic frame from a spatial model written in
    Fermi-XML format.

    If the value of spatial extension, sigma is required, it will also be fetched.
    """
    if not sigma:
        sigma = None

    for par_ in spatial_params:
        match par_["@name"]:
            case "RA":
                lon_0 = f"{par_['@value']} deg"
            case "DEC":
                lat_0 = f"{par_['@value']} deg"
            case "Sigma":
                sigma = f"{par_['@value']} deg"
    fk5_frame = SkyCoord(
        lon_0,
        lat_0,
        frame="fk5",
    )

    return fk5_frame.transform_to("galactic"), sigma


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
        match xml_spatial_model["@type"]:
            case "SkyDirFunction":
                gal_frame, _ = fetch_spatial_galactic_frame(spatial_pars, sigma=False)
                spatial_model = SPATIAL_MODEL_REGISTRY.get_cls("PointSpatialModel")().from_position(gal_frame)

            case "SpatialMap":
                file_name = xml_spatial_model["@file"].split("/")[-1]
                file_path = aux_path / f"Templates/{file_name}"

                spatial_map = Map.read(file_path)
                spatial_map = spatial_map.copy(unit="sr^-1")

                spatial_model = SPATIAL_MODEL_REGISTRY.get_cls("TemplateSpatialModel")(
                    spatial_map, filename=file_path
                )

            case "RadialGaussian":
                gal_frame, sigma = fetch_spatial_galactic_frame(spatial_pars, sigma=True)
                spatial_model = SPATIAL_MODEL_REGISTRY.get_cls("GaussianSpatialModel")(
                    lon_0=gal_frame.l, lat_0=gal_frame.b, sigma=sigma, frame="galactic"
                )

    return spatial_model
