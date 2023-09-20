Additional Support of Models input
==================================

:ref:`models-intro` gives a brief overview on including the Models to the DL4 datasets.

The list of associated Models can be provided by -

#. Using models defined in Gammapy
    The list of Models used in Gammapy can be seen in
    `Model Gallery <https://docs.gammapy.org/1.1/user-guide/model-gallery/index.html>`_.

    Additional models following the Gammapy standards, are defined in
    :doc:`_api_docs/data/target/data_target_b`, and to use these models,
    one needs to invoke that module.

#. Using a list of models written in a different way than the Gammapy standard
    The module :class:`~asgardpy.gammapy.interoperate_models` is used to read such models,
    for e.g. XML model definitions used by Fermi-LAT.

    A test covering all of these models is included in asgardpy, using a test
    xml file, included in the additional test data.

#. Using a Catalog available in Gammapy
    This is done by adding information in :class:`~asgardpy.data.target.Target.use_catalog`.
    The list of available catalogs in Gammapy is documented at
    `Source Catalogs <https://docs.gammapy.org/1.1/user-guide/catalog.html>`_

#. Using FoV Background Model as defined in Gammapy
    To add a default Gammapy `FoVBackgroundModel` to the 3D dataset, use
    :class:`~asgardpy.data.target.Target.add_fov_bkg_model` ``= True``.

    One can also include it in the config file by defining its spectral and/or
    spatial model components.
