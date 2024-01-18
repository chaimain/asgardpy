"""
Functions for reading different Models type objects with Gammapy
objects.
"""
import xmltodict
from gammapy.maps import Map
from gammapy.modeling.models import (
    SPATIAL_MODEL_REGISTRY,
    SPECTRAL_MODEL_REGISTRY,
    SkyModel,
    create_fermi_isotropic_diffuse_model,
)

from asgardpy.data.target import (
    add_ebl_model_from_config,
    read_models_from_asgardpy_config,
)
from asgardpy.gammapy.interoperate_models import (
    get_gammapy_spectral_model,
    xml_spatial_model_to_gammapy,
    xml_spectral_model_to_gammapy,
)

__all__ = [
    "create_gal_diffuse_skymodel",
    "create_iso_diffuse_skymodel",
    "create_source_skymodel",
    "read_fermi_xml_models_list",
    "update_aux_info_from_fermi_xml",
]


def read_fermi_xml_models_list(
    list_source_models, dl3_aux_path, xml_file, diffuse_models, asgardpy_target_config=None
):
    """
    Read from the Fermi-XML file to enlist the various objects and get their
    SkyModels.

    Parameters
    ----------
    list_source_models: list
        List of source models to be filled.
    dl3_aux_path: str
        Path location of the DL3 auxiliary files for reading Spatial Models
        from separate files.
    xml_file: str
        Path of the Fermi-XML file to be read.
    diffuse_models: dict
        Dict containing diffuse models' filenames and instrument key.
    asgardpy_target_config: `AsgardpyConfig`
        Config section containing the Target source information.

    Returns
    -------
    list_source_models: list
        List of filled source models.
    diffuse_models: dict
        Dict containing diffuse models objects replacing the filenames.
    """
    with open(xml_file, encoding="utf-8") as file:
        xml_data = xmltodict.parse(file.read())["source_library"]["source"]

    is_target_source = False
    for source_info in xml_data:
        match source_info["@name"]:
            # Nomenclature as per enrico and fermipy
            case "IsoDiffModel" | "isodiff":
                diffuse_models["iso_diffuse"] = create_iso_diffuse_skymodel(
                    diffuse_models["iso_diffuse"], diffuse_models["key_name"]
                )
                source_info = diffuse_models["iso_diffuse"]

            case "GalDiffModel" | "galdiff":
                diffuse_models["gal_diffuse"], diffuse_models["gal_diffuse_map"] = create_gal_diffuse_skymodel(
                    diffuse_models["gal_diffuse"]
                )
                source_info = diffuse_models["gal_diffuse"]

            case _:
                source_info, is_target_source = create_source_skymodel(
                    source_info,
                    dl3_aux_path,
                    base_model_type="Fermi-XML",
                    asgardpy_target_config=asgardpy_target_config,
                )

        if is_target_source:
            list_source_models.insert(0, source_info)
            is_target_source = False  # To not let it repeat
        else:
            list_source_models.append(source_info)

    return list_source_models, diffuse_models


def update_aux_info_from_fermi_xml(
    diffuse_models_file_names_dict, xml_file, fetch_iso_diff=False, fetch_gal_diff=False
):
    """
    When no glob_search patterns on axuillary files are provided, fetch
    them from the XML file and update the dict containing diffuse models' file
    names.

    Parameters
    ----------
    diffuse_models_file_names_dict: `dict`
        Dict containing the information on the DL3 files input.
    xml_file: str
        Path of the Fermi-XML file to be read.
    fetch_iso_diff: bool
        Boolean to get the information of the Isotropic diffuse model from the
        Fermi-XML file
    fetch_gal_diff: bool
        Boolean to get the information of the Galactic diffuse model from the
        Fermi-XML file

    Returns
    -------
    diffuse_models_file_names_dict: `dict`
        Dict containing the updated information on the DL3 files input.
    """
    with open(xml_file) as file:
        data = xmltodict.parse(file.read())["source_library"]["source"]

    for source in data:
        match source["@name"]:
            case "IsoDiffModel" | "isodiff":
                if fetch_iso_diff:
                    file_path = source["spectrum"]["@file"]
                    file_name = file_path.split("/")[-1]
                    diffuse_models_file_names_dict["iso_diffuse"] = file_name

            case "GalDiffModel" | "galdiff":
                if fetch_gal_diff:
                    file_path = source["spatialModel"]["@file"]
                    file_name = file_path.split("/")[-1]
                    diffuse_models_file_names_dict["gal_diffuse"] = file_name

    return diffuse_models_file_names_dict


def get_target_model_from_config(source_name, asgardpy_target_config):
    """
    Function to get the model of the target source from AsgardpyConfig.
    """
    spectral_model = None
    is_source_target = False

    # If Target source model's spectral component is to be taken from Config
    # and not from 3D dataset.
    if asgardpy_target_config:
        source_name_check = source_name.replace("_", "").replace(" ", "")
        target_check = asgardpy_target_config.source_name.replace("_", "").replace(" ", "")

        if source_name_check == target_check:
            source_name = asgardpy_target_config.source_name
            is_source_target = True

            # Only taking the spectral model information right now.
            if not asgardpy_target_config.from_3d:
                models_ = read_models_from_asgardpy_config(asgardpy_target_config)
                spectral_model = models_[0].spectral_model

    return source_name, spectral_model, is_source_target


def create_source_skymodel(source_info, dl3_aux_path, base_model_type="Fermi-XML", asgardpy_target_config=None):
    """
    Build SkyModels from given base model information.

    If AsgardpyConfig section of the target is provided for the target
    source information, it will be used to check if the target `source_name`
    is provided in the base_model file. If it exists, then check if the model
    information is to be read from AsgardpyConfig using `from_3d` boolean value.
    Also, if EBL model information is provided in the AsgardpyConfig, it will
    be added to the SkyModel object.

    Parameters
    ----------
    source_info: dict
        Dictionary containing the source models information from XML file.
    dl3_aux_path: str
        Path location of the DL3 auxiliary files for reading Spatial Models
        from separate files.
    base_model_type: str
        Name indicating the model format used to read the skymodels from.
    asgardpy_target_config: `AsgardpyConfig`
        Config section containing the Target source information.

    Returns
    -------
    source_sky_model: `gammapy.modeling.SkyModel`
        SkyModels object for the given source information.
    is_source_target: bool
        Boolean to check if the Models belong to the target source.
    """
    if base_model_type == "Fermi-XML":
        source_name = source_info["@name"]
        spectrum_type = source_info["spectrum"]["@type"]
        spectrum_params = source_info["spectrum"]["parameter"]

        # initialized to check for the case if target spectral model information
        # is to be taken from the Config
        spectral_model = None

        # Check if target_source file exists
        is_source_target = False
        ebl_atten = False

        source_name, spectral_model, is_source_target = get_target_model_from_config(
            source_name, asgardpy_target_config
        )

        if spectral_model is None:
            # Define the Spectral Model type for Gammapy
            spectral_model, ebl_atten = get_gammapy_spectral_model(
                spectrum_type,
                ebl_atten,
                base_model_type,
            )
            spectrum_type = spectrum_type.split("EblAtten::")[-1]

            # Read the parameter values from XML file to create SpectralModel
            params_list = xml_spectral_model_to_gammapy(
                spectrum_params,
                spectrum_type,
                is_target=is_source_target,
                keep_sign=ebl_atten,
                base_model_type=base_model_type,
            )

            for param_ in params_list:
                setattr(spectral_model, param_.name, param_)

            if asgardpy_target_config:
                model_config = asgardpy_target_config.components[0]
                ebl_absorption_included = model_config.spectral.ebl_abs.reference != ""

                if is_source_target and ebl_absorption_included:
                    spectral_model = add_ebl_model_from_config(spectral_model, model_config)

        # Reading Spatial model from the XML file
        spatial_model = xml_spatial_model_to_gammapy(dl3_aux_path, source_info["spatialModel"], base_model_type)

        spatial_model.freeze()
        source_sky_model = SkyModel(
            spectral_model=spectral_model,
            spatial_model=spatial_model,
            name=source_name,
        )

    return source_sky_model, is_source_target


def create_iso_diffuse_skymodel(iso_file, key_name):
    """
    Create a SkyModel of the Fermi Isotropic Diffuse Model and assigning
    name as per the observation key. If there are no distinct key types of
    files, the value is None.

    Parameters
    ----------
    iso_file: str
        Path to the isotropic diffuse model file
    key: str
        Instrument key-name, if exists

    Returns
    -------
    diff_iso: `gammapy.modeling.SkyModel`
        SkyModel object of the isotropic diffuse model.
    """
    diff_iso = create_fermi_isotropic_diffuse_model(filename=iso_file, interp_kwargs={"fill_value": None})

    if key_name:
        diff_iso._name = f"{diff_iso.name}-{key_name}"

    # Parameters' limits
    diff_iso.spectral_model.model1.parameters[0].min = 0.001
    diff_iso.spectral_model.model1.parameters[0].max = 10
    diff_iso.spectral_model.model2.parameters[0].min = 0
    diff_iso.spectral_model.model2.parameters[0].max = 10

    return diff_iso


def create_gal_diffuse_skymodel(diff_gal_file):
    """
    Create SkyModel of the Diffuse Galactic sources either by reading a file or
    by using a provided Map object.

    Parameters
    ----------
    diff_gal_file: Path, `gammapy.maps.Map`
        Path to the isotropic diffuse model file, or a Map object already read
        from a file.

    Returns
    -------
    model: `gammapy.modeling.SkyModel`
        SkyModel object read from a given galactic diffuse model file.
    diff_gal: `gammapy.maps.Map`
        Map object of the galactic diffuse model
    """
    if not isinstance(diff_gal_file, Map):
        # assuming it is Path or str type
        diff_gal_filename = diff_gal_file
        diff_gal = Map.read(diff_gal_file)
        diff_gal.meta["filename"] = diff_gal_filename
    else:
        diff_gal = diff_gal_file

    template_diffuse = SPATIAL_MODEL_REGISTRY.get_cls("TemplateSpatialModel")(
        diff_gal, normalize=False, filename=diff_gal.meta["filename"]
    )
    model = SkyModel(
        spectral_model=SPECTRAL_MODEL_REGISTRY.get_cls("PowerLawNormSpectralModel")(),
        spatial_model=template_diffuse,
        name="diffuse-iem",
    )
    model.parameters["norm"].min = 0
    model.parameters["norm"].max = 10
    model.parameters["norm"].frozen = False

    return model, diff_gal
