Asgardpy 0.4.4 (2024-07-12)
===========================

Hotfix for 0.4.3 by including commits from [`#161 <https://github.com/chaimain/asgardpy/pull/161>`__], [`#171 <https://github.com/chaimain/asgardpy/pull/171>`__] and [`#174 <https://github.com/chaimain/asgardpy/pull/174>`__].


API Changes
-----------


Bug Fixes
---------


New Features
------------


Maintenance
-----------

Asgardpy 0.4.3 (2024-01-27)
===========================


API Changes
-----------


Bug Fixes
---------


New Features
------------

- Separate the functionality of reading from a Fermi-XML model file into a distinct function in gammapy module. Completes the Issue [`#153 <https://github.com/chaimain/asgardpy/issues/153>`_]. [`#155 <https://github.com/chaimain/asgardpy/pull/155>`__]

- Add a temporary stats function to fetch pivot energy for any spectral model. This function will be present in the upcoming release of Gammapy v1.2.

  Also include Windows OS in CI tests. [`#160 <https://github.com/chaimain/asgardpy/pull/160>`__]


Maintenance
-----------

- Show Towncrier created fragments for changelog drafts before release of a new project version. [`#151 <https://github.com/chaimain/asgardpy/pull/151>`__]

- Adapt tox env and start using pre-commit hooks. Have CI also run on Mac OS. Completes the remaining tasks in the Issue [`#127 <https://github.com/chaimain/asgardpy/issues/127>`_]. [`#154 <https://github.com/chaimain/asgardpy/pull/154>`__]

- Adapt code suggestions from Scientific Python Development Guide and remove some coverage redundancies. [`#156 <https://github.com/chaimain/asgardpy/pull/156>`__]

- Adapt code suggestions from Scrutinizer Code Quality scan. Use match-case instead of list of if-elif-else condition blocks. [`#157 <https://github.com/chaimain/asgardpy/pull/157>`__]

- Limit scipy version until gammapy v1.2 fixes its dependency. [`#158 <https://github.com/chaimain/asgardpy/pull/158>`__]

Asgardpy 0.4.2 (2023-11-20)
===========================


API Changes
-----------


Bug Fixes
---------

- Apply Bug fix on the definition of ``TimeIntervals`` config and usage in ``ObservationsConfig`` by changing it to just be a dict type and letting the ``ObservationsConfig`` have the ``obs_time`` as list of ``TimeIntervalsType``. Fixes the issue [`#144 <https://github.com/chaimain/asgardpy/issues/144>`_]. [`#145 <https://github.com/chaimain/asgardpy/pull/145>`__]


New Features
------------

- Fixed project descriptions by correcting License id, changelog file link also added some more test coverage. [`#137 <https://github.com/chaimain/asgardpy/pull/137>`__]

- Expand test coverage and add a description for the additional zipped Fermi-LAT data. [`#138 <https://github.com/chaimain/asgardpy/pull/138>`__]

- Extend capacity to have more than 1 3D datasets, with any of them being able to add an extra FoV Background Model, to show the full support of the pipeline. [`#139 <https://github.com/chaimain/asgardpy/pull/139>`__]

- Implement towncrier for maintaining PR changes as new fragments. [`#141 <https://github.com/chaimain/asgardpy/pull/141>`__]

- Adapt the basic Time format to use any format and list of intervals as part of ``BaseConfig``.
  Remove redundant classes, adapt tests and use ``TimeIntervalsType`` as the main Time input type. [`#142 <https://github.com/chaimain/asgardpy/pull/142>`__]

- Extend the functionality of ``recursive_merge_dicts`` when comparing 2 lists of values with different lengths. [`#143 <https://github.com/chaimain/asgardpy/pull/143>`__]


Maintenance
-----------

Asgardpy v0.4.1 (2023-09-27)
============================


Bug fixes
---------

- Correct the calculation of Goodness of Fit stats as described in the Issue
  [`#130 <https://github.com/chaimain/asgardpy/issues/130>`_]. Replaces the
  usage of Gammapy's ``CountsStatistic`` functions with those in
  ``gammapy.stats.fit_statistics`` module, to evaluate the best fit (Observed)
  statistic and perfect fit (Expected) statistic for all types of Datasets used.

  Update the various notebooks, scripts and documentation accordingly.

  Update test coverage on ``analysis`` and ``data`` modules, along with adding
  copies of diffuse model files for the additional Fermi-LAT data to use name ids
  used by ``fermipy`` and ``enrico``.
  [`#132 <https://github.com/chaimain/asgardpy/pull/132>`_]

- Replaced dynamic versioning to go back to static versioning as in the parent
  package template, before trying a proper resolution of Issue
  [`#135 <https://github.com/chaimain/asgardpy/issues/135>`_].
  [`#136 <https://github.com/chaimain/asgardpy/pull/136>`_]

- Update documentation to have separate pages for describing additional support
  for asgardpy inputs, from the standard ones, mentioned in the overview page.

  Also updates test coverage for ``data.target`` and ``io`` modules.
  [`#134 <https://github.com/chaimain/asgardpy/pull/134>`_]

- Update test coverage on reading different XML models for Fermi-LAT data as
  supported by ``gammapy`` module of asgardpy and move the zipped additional
  data to ``dev`` folder. [`#133 <https://github.com/chaimain/asgardpy/pull/133>`_]

- Updated pytests to follow proper code style with proper assert statements as
  mentioned in one of the 3 tasks in the Issue
  [`#127 <https://github.com/chaimain/asgardpy/issues/127>`_].
  [`#131 <https://github.com/chaimain/asgardpy/pull/131>`_]


Maintenance
-----------

- Restructure the package to follow some PEP guidelines as described in
  Issue [`#125 <https://github.com/chaimain/asgardpy/issues/125>`_].

  Add a separate Citation file, move codespell_ignore_words file to ``dev`` folder,
  Contributing file to ``.github`` folder. Start to use dynamic versioning.
  [`#126 <https://github.com/chaimain/asgardpy/pull/126>`_]


Asgardpy v0.4.0 (2023-08-31)
============================


New Features
------------

- Add more tests to increase coverage and perform some basic clean-up of the
  tests. Optimize tests using flux-points analysis step and reading catalog data.
  [`#122 <https://github.com/chaimain/asgardpy/pull/122>`_]

- Add tests to cover some of the Gammapy tutorial examples, like the MWL joint
  fitting of different datasets, including reading from a ``FluxPointsDataset``
  object. and fix the relevant code accordingly in the various modules.

  Have a generalized ``ModelComponent`` to include ``SkyModel`` and ``FoVBackgroundModel``
  for reading Gammapy Models objects.

  Have a test for checking model preference script, instead of filling the
  jupyter notebook. [`#121 <https://github.com/chaimain/asgardpy/pull/121>`_]

- Add tests to various modules and extend support for CI test runs by
  increasing swapfile size of the system. Also allow conftest to check for
  existing ``gammapy-data`` in the system before running any tests with data.

  Update notebooks after running with the test data, the overall documentation
  about the tests and pytest options in pyproject.toml file.

  Fully resolves Issue [`#55 <https://github.com/chaimain/asgardpy/issues/55>`_].
  [`#120 <https://github.com/chaimain/asgardpy/pull/120>`_]


Maintenance
-----------

- Perform general cleanup and fix minor pending issues.
  [`#124 <https://github.com/chaimain/asgardpy/pull/124>`_]

- Add Code of Conduct to the package and Codecov support in CI.
  [`#113 <https://github.com/chaimain/asgardpy/pull/113>`_]


API Changes
-----------

- Restructure tests to reduce overall test time.
  [`#123 <https://github.com/chaimain/asgardpy/pull/123>`_]

- Restructure ``io`` by replacing current sub-module with ``input_dl3`` and
  ``io_dl4`` modules, containing functions related with DL3 and DL4 files.

  Adds public test data, as an addition to the existing Gammapy test data, by
  having a zip compressed file, containing ``Fermi-LAT`` data generated with
  ``fermipy`` for Crab Nebula observations. The CI checks for the presence of
  downloaded ``gammapy-data`` and it being saved in the environ path variable
  of the system, and only then unzips the additional data, in the same location,
  in a folder named ``fermipy-crab``.

  Using these updated test data, starts building simple pytests by using test
  template config files in ``tests`` module. The additional support of DL4 data
  input, will help in replicating some tests done in Gammapy, to check the
  additional support by Asgardpy.

  See Issue [`#55 <https://github.com/chaimain/asgardpy/issues/55>`_] for more
  details, as this PR, resolves yet another aspect of the Issue.
  [`#114 <https://github.com/chaimain/asgardpy/pull/114>`_]


Asgardpy v0.3.6 (2023-08-05)
============================


API Changes
-----------

- Restructure statistics functions to be part of a separate ``stats`` module.
  Collect relevant information for estimating the goodness of fit stats, in the
  ``instrument_spectral_info`` dict variable, to be used only when the ``fit``
  analysis step is completed. Update computation of fit statistics using
  internal Gammapy functions to get appropriate results.
  [`#103 <https://github.com/chaimain/asgardpy/pull/103>`_]

- Restructure pipeline to prepare to use public test data for resolving Issue
  [`#55 <https://github.com/chaimain/asgardpy/issues/55>`_].

  Have a distinct module ``gammapy`` containing all functions for
  interoperatibility of other data formats with Gammapy format, for example,
  the XML model definition used by Fermi-LAT. Generalize this usage for any
  other model definition for future additional support. Update docstring with
  Fermi-LAT model functions NOT supported by this function for future tracking.

  Added function to read from a Gammapy ``AnalysisConfig`` file, into an
  ``AsgardpyConfig`` file for increased support.

  Add support for reading ``FoVBackgroundModel`` from config file.

  Move model template files into a separate folder.
  [`#110 <https://github.com/chaimain/asgardpy/pull/110>`_]


Bug Fixes
---------

- Improve logging as per the Issue [`#39 <https://github.com/chaimain/asgardpy/issues/39>`_]

  From recommendations of pylint code style, update pending docstrings of
  various functions and modules, fix logging strings. Also include flake8 and
  codespell settings in setup.cfg file and include codespell check in CI.
  [`#102 <https://github.com/chaimain/asgardpy/pull/102>`_]

- Fix estimation of Goodness of Fit statistics by removing the extra function
  on evaluating Test Statistic for Null Hypothesis and combining it into a new
  common function ``get_ts_target``, to get the required TS values of both Null
  and Alternate Hypotheses, only for the region (binned coordinates) of the
  target source.

  Separate the counting of the total degrees of freedom, into total number
  of reco energy bins used and the number of free model parameters.
  [`#106 <https://github.com/chaimain/asgardpy/pull/106>`_]

- Resolve the issue of circular imports by restructuring analysis module to
  have separate scripts with ``AnalysisStepBase`` and ``AnalysisStep`` classes.

  Moved ``SkyPositionConfig`` to ``asgardpy.base.geom`` module and using imports
  from specific sub-modules when required.
  [`#107 <https://github.com/chaimain/asgardpy/pull/107>`_]

- Fix reading of ``models_file`` with the correct process.
  [`#112 <https://github.com/chaimain/asgardpy/pull/112>`_]


Maintenance
-----------

- Add codespell to dev-requirements.
  [`#104 <https://github.com/chaimain/asgardpy/pull/104>`_]

- Compress and update sphinx docs, by having documentation pages based on
  distinct modules.
  [`#105 <https://github.com/chaimain/asgardpy/pull/105>`_]

- Update python dependency to 3.11, added OpenSSF Best Practices badge in README
  and a dedicated Issue Tracker link in documentation.
  [`#109 <https://github.com/chaimain/asgardpy/pull/109>`_]


Asgardpy v0.3.5 (2023-07-17)
============================


API Changes
-----------

- Restructure pipeline to regroup common functions, for base geometry and data
  reduction for GADF-based DL3 files for 1D and 3D dataset. Use ``DatasetsMaker``
  for supporting parallel processing of DL4 dataset generation.
  See Issue [`#85 <https://github.com/chaimain/asgardpy/issues/85>`_]

  Update support for Ring and FoV Background Makers, and have a separate common
  function for creating exclusion masks for datasets.

  Keep GADF-based DL3 input as default priority for generating 3D datasets.

  Have a simple test for importing main Asgardpy classes, and a simple script
  to run all Analysis steps of a given AsgardpyConfig file.

  Update basic docstrings of various functions and classes.
  [`#94 <https://github.com/chaimain/asgardpy/pull/94>`_]

- Restructure pipeline for better handling of model association, by adding
  support to use catalog data for getting the list of source models and for
  creating exclusion regions in the Field of View, using ``FoVBackgroundModel``,
  renaming the variable, ``extended`` in ``target`` config section to
  ``add_fov_bkg_model``, moving the application of exclusion mask onto the list
  of models to the ``set_models`` function and update these into the
  documentation page. Completing the remaining task in the
  Issue [`#85 <https://github.com/chaimain/asgardpy/issues/85>`_]

  Group the processing of Analysis Steps into DL3 to DL4 and DL4 to DL5 stages.
  [`#98 <https://github.com/chaimain/asgardpy/pull/98>`_]

- Add a single function to get the chi2 and p-value of a given test statistic
  and degrees of freedom and generalize other stat functions, to use more specific
  variables. [`#101 <https://github.com/chaimain/asgardpy/pull/101>`_]


Bug Fixes
---------

- Update documentation with new workflow image and the notebooks.
  [`#97 <https://github.com/chaimain/asgardpy/pull/97>`_]

- General clean-up and addition of docstrings to various Configs.
  [`#99 <https://github.com/chaimain/asgardpy/pull/99>`_]

- Fix the Changelog to be more descriptive.
  [`#100 <https://github.com/chaimain/asgardpy/pull/100>`_]


Maintenance
-----------

- Update documentation with citation link using Zenodo DOI and add the badge in
  README. [`#95 <https://github.com/chaimain/asgardpy/pull/95>`_]


Asgardpy v0.3.4 (2023-07-02)
============================


New Features
------------

- Add script to get most preferred spectral model fit based on the existing
  notebook.
  Also add extra supporting functions to get any model template config files,
  have a check on statistically preferred models based on likelihood ratio test
  and Akaike Information Criterion and updating the notebook accordingly.
  [`#87 <https://github.com/chaimain/asgardpy/pull/87>`_]


API Changes
-----------

- Combine the various Sky Position configs into a single ``SkyPositionConfig``,
  with the information of the coordinate frame, longitude, latitude and
  angular radius, where for defining point source, the angular radius has a
  default value of 0 degree. [`#88 <https://github.com/chaimain/asgardpy/pull/88>`_]


Bug Fixes
---------

- Update documentation by replacing the model parameter renaming table and the
  extended support added in previous PR. [`#89 <https://github.com/chaimain/asgardpy/pull/89>`_]

- Update with usage of common multiprocessing with Gammapy for generating DL4
  datasets and Flux Points Estimation. [`#90 <https://github.com/chaimain/asgardpy/pull/90>`_]


Maintenance
-----------

- Constrain pydantic and autodoc-pydantic versions until corresponding updates
  are made in Gammapy. [`#92 <https://github.com/chaimain/asgardpy/pull/92>`_]


Asgardpy v0.3.3 (2023-06-20)
============================


Bug Fixes
---------

- Fix sphinx documentation build issue by updating the readthedocs config file
  with build information. [`#78 <https://github.com/chaimain/asgardpy/pull/78>`_]

- Try to fix sphinx documentation build issue by removing the deprecated
  ``python.version`` information. [`#79 <https://github.com/chaimain/asgardpy/pull/79>`_]

- Update Sphinx documentation for all modules, use ``autodoc_pydantic``, divide
  the documentation of ``asgardpy.data.target`` into 2 separate pages and fix
  missing functions in the documentation.

  Update the template config file and have a copy for documentation.

  Remove redundant Analysis steps from the list.
  [`#81 <https://github.com/chaimain/asgardpy/pull/81>`_]


Maintenance
-----------

- Extend support to Gammapy v1.1 by adding parallel processing support and
  update general dependency requirement conditions.
  [`#84 <https://github.com/chaimain/asgardpy/pull/84>`_]


Asgardpy v0.3.2 (2023-04-28)
============================


New Features
------------

- Add the custom spectral models to the Gammapy registry while using Asgardpy.
  [`#77 <https://github.com/chaimain/asgardpy/pull/77>`_]


Asgardpy v0.3.1 (2023-04-28)
============================


Maintenance
-----------

- Remove support of python 3.8. [`#76 <https://github.com/chaimain/asgardpy/pull/76>`_]


Asgardpy v0.3.0 (2023-04-28)
============================


Bug Fixes
---------

- Update ``config`` module with a function to perform recursive merging, see
  Issue [`#71 <https://github.com/chaimain/asgardpy/issues/71>`_]. This is
  used when the model config is provided as a separate file, which does not
  contain a model name. Examples of such files are also created for a variety
  of spectral models.

  Using the multiple available options for spectral models, one can check for a
  statistically preferred model for a given dataset, by using methods like
  likelihood ratio test, Akaike Information Criterion, etc. A notebook is added
  to demonstrate this procedure.

  Also fixed a URL link of a badge in README.
  [`#72 <https://github.com/chaimain/asgardpy/pull/72>`_]

- Update README with more description and a Build status badge.

  Update general documentation, change the description of ``asgardpy`` from a
  ``package`` to a ``pipeline`` and add a setup.cfg file with the general
  description of asgardpy. [`#73 <https://github.com/chaimain/asgardpy/pull/73>`_]

- Fix Safe Mask reduction code for 1D Dataset and add a custom Spectral Model of
  Broken Power Law with ``index_diff`` as a parameter, to get the second power
  law index, with respect to the index of the first one.
  [`#74 <https://github.com/chaimain/asgardpy/pull/74>`_]


API Changes
-----------

- Restructure the pipeline to have a distinct ``base`` module, to avoid circular
  imports issue and shifting the modules and classes for defining the base
  class for Analysis Steps, base geometry of datasets and dataset reduction
  methods.

  Sort the imports for better coding practice.

  Remove redundant ``glob_dict_std`` variable in ``io`` module.

  Update documentation and notebooks accordingly.
  [`#75 <https://github.com/chaimain/asgardpy/pull/75>`_]


Asgardpy v0.2.0 (2023-04-19)
============================


Bug Fixes
---------

- Update documentation with correct URL paths, providing proper descriptions of
  various modules and the main working of the pipeline and some formatting
  corrections.
  [`#54 <https://github.com/chaimain/asgardpy/pull/54>`_]

- Update notebooks. [`#69 <https://github.com/chaimain/asgardpy/pull/69>`_]


New Features
------------

- Build some custom SpectralModel classes.
  [`#59 <https://github.com/chaimain/asgardpy/pull/59>`_]

- Add support for common data types for different instruments by have a standard
  ``dl3_type`` as ``gadf-dl3`` instead of instrument specific like ``lst-1`` and
  improve the conditions for checking its different values. See Issue
  [`#34 <https://github.com/chaimain/asgardpy/issues/34>`_] for more details.
  [`#65 <https://github.com/chaimain/asgardpy/pull/65>`_]

- Add support for selecting various spectral model parameters in a given Field
  of View, by generalizing the function ``apply_selection_mask_to_models`` in
  the ``asgardpy.data.target`` module.
  [`#67 <https://github.com/chaimain/asgardpy/pull/67>`_]

API Changes
-----------

- Remove dependency of hard-coded Fermi-LAT files structure to move towards a
  generalized 3D datasets input.

  Expand the scope of creating exclusion mask for 1D dataset.
  [`#56 <https://github.com/chaimain/asgardpy/pull/56>`_]

- Adding the possibility to use a separate yaml file for providing Target source
  model information and reading the file pathas ``models_file`` variable.

  Fix some variable names to be the same as used in Gammapy and for moving
  towards generalizing the pipeline.

  Separate documentation of each sub-module of ``asgardpy.data`` module.
  [`#57 <https://github.com/chaimain/asgardpy/pull/57>`_]

- Adding support of reading EBL models from fits files.

  Also adds to the index page of the documentation, an introduction to the
  package and moving the Development links to the sidebar.
  [`#58 <https://github.com/chaimain/asgardpy/pull/58>`_]

- Incorporate input of Fermi-LAT files, generated with fermipy into
  ``Dataset3DGeneration`` function by generalizing the process of defining the
  base geometry of a Counts Map, reading diffuse model names from the XML file
  and some re-arrangement of the general procedure.

  Add functions to read spectral and spatial model information from different
  formats to the standard Gammapy format, and improve the ``asgardpy.data.target``
  module in general.

  Rename some variables in ``data`` and ``io`` modules accordingly.
  [`#61 <https://github.com/chaimain/asgardpy/pull/61>`_]

- Remove features from the package that are not essential and can be used with
  Gammapy alone. These are the Analysis steps of ``light-curve-estimator``,
  ``excess-map``, ``DL4Files`` class for writing data products to separate files
  and ``asgardpy.utils`` module, containing basic plot functions. These are
  listed in the Issue [`#60 <https://github.com/chaimain/asgardpy/issues/60>`_].
  [`#62 <https://github.com/chaimain/asgardpy/pull/62>`_]

- Generalize the usage of ``GeomConfig`` for both type of Datasets. Let user
  define non-spatial axes to define the base geometry, currently being only of
  Energy, differentiating from the energy parameters used for generating SEDs in
  ``flux-points`` Analysis Step, using ``spectral_energy_range`` component. See
  connected Issue [`#28 <https://github.com/chaimain/asgardpy/issues/28>`_].

  Generalize mapping of Models from different format to Gammapy-compliant format,
  by having two separate functions, ``params_renaming_to_gammapy`` and
  ``params_rescale_to_gammapy`` for Spectral Model. See Issue
  [`#52 <https://github.com/chaimain/asgardpy/issues/52>`_] for more detail.

  Extend support to map ``PLSuperExpCutoff2`` spectral model of Fermi-XML type and
  ``GaussianSpatialModel``.

  Add images in the documentation to show the workflow of the package and the
  model parameters mapping from Fermi-XML type to Gammapy type.

  Have the option to read 3D dataset information when no distinct ``key`` names
  are provided.
  [`#64 <https://github.com/chaimain/asgardpy/pull/64>`_]

- Generalize reading energy axes by using a distinct function ``get_energy_axis``
  in ``asgardpy.data.geom`` module. Let ``spectral_energy_range`` be of
  ``MapAxesConfig`` type for more uniform reading of this information. Also
  allow for providing custom energy bin edges for this variable, to be used to
  create SEDs. [`#68 <https://github.com/chaimain/asgardpy/pull/68>`_]

- Remove GTI selections from 3D datasets, as at least for Fermi-LAT datasets,
  the files are produced for a select set of GTI time intervals amongst other
  selections and the various files produced, are exclusive for these selections.

  GADF-DL3 type of 1D dataset can still have GTI selection option, but it should
  correspond to the GTI interval for the Fermi-LAT data.
  [`#70 <https://github.com/chaimain/asgardpy/pull/70>`_]


Asgardpy v0.1 (2023-02-16)
============================


New Features
------------

- Start adding requirements and dependencies and use a minimum python version
  of 3.8 instead of 3.7.
  [`#6 <https://github.com/chaimain/asgardpy/pull/6>`_]

- Start with some I/O classes and functions for DL3 and DL4 files in a ``io``
  module. [`#7 <https://github.com/chaimain/asgardpy/pull/7>`_]

- Start entering Fit and plot functions in ``analysis`` module.
  [`#11 <https://github.com/chaimain/asgardpy/pull/11>`_]

- Proposal for the initial template for the pipeline to perform the following
  steps,

  1. Read the various instrument DL3 files

  2. Perform any and all data reductions

  3. Generate Datasets for each instrument

  4. Pass the list of all such Datasets to the Gammapy Fit function to get the
  best-fit model

  The other functionalities can be left to the user to perform without using
  asgardpy. [`#15 <https://github.com/chaimain/asgardpy/pull/15>`_]

- Build further the pipeline structure, by generalizing the dataset production
  as 1D or 3D, let the ``DL3Files`` class be the base class for all DL3 Files
  input.

  Also include a release drafter template in .github folder.
  [`#16 <https://github.com/chaimain/asgardpy/pull/16>`_]

- Include the release-drafter in github CI workflow.
  [`#18 <https://github.com/chaimain/asgardpy/pull/18>`_]

- Begin preparations for adding workable scripts.
  Restructure classes of Analysis Steps for creating 1/3 D datasets to only
  have a single Analysis Step to be run for for each type of dataset and to
  have the various components for data selection, reduction and creation of the
  DL4 dataset, as a separate class which will be called when running the
  particular ``AnalysisStep``.

  Rename the module responsible for the ``AnalysisSteps`` of working with the
  DL4 datasets, to Fit Models, Flux Points and Light Curve Estimation, to
  ``asgardpy.data.dl4``.

  Add more configuration options for defining Background Reduction Makers, using
  currently only "reflected" and "wobble" ``RegionsFinder`` methods.

  Move the functions for Models assignment into ``asgardpy.data.target`` module.

  Improve the method of DL3 files config input in the ``asgardpy.io`` module.

  Add ``AsgardpyAnalysis`` class that handles running of all Analysis Steps,
  based on the Gammapy HLI ``Analysis`` class.
  [`#19 <https://github.com/chaimain/asgardpy/pull/19>`_]

- Improve reading Models and assigning them to DL4 datasets, to be closer to the
  functionality of Gammapy. Move all such functions to the ``target`` module.
  See Issue [`#29 <https://github.com/chaimain/asgardpy/issues/29>`_] for more
  details.

  Introduce a separate function to read Gammapy models from the ``AsgardpyConfig``
  information and also to convert the Models information from XML model of
  FermiTools to Gammapy standard.

  Have a new object of ``AsgardpyAnalysis`` as ``final_model`` to make it
  easier to read list of models before and after assignment to DL4 datasets.
  [`#31 <https://github.com/chaimain/asgardpy/pull/31>`_]

- Adding plotting functions into a separate module ``asgardpy.utils`` and
  update the AnalaysisStep ``flux-points`` by using constant number of energy
  bins per decade for each dataset, but keeping the range within each dataset's
  energy axes. [`#32 <https://github.com/chaimain/asgardpy/pull/32>`_]

- Start using GTI time intervals for creating DL4 datasets and ``light-curve``
  analysis step. See Issue [`#30 <https://github.com/chaimain/asgardpy/issues/30>`_]
  for more details. [`#35 <https://github.com/chaimain/asgardpy/pull/35>`_]

- Start adding example notebooks and starting with a single notebook for the
  full analysis. [`#37 <https://github.com/chaimain/asgardpy/pull/37>`_]

- Addition of instrument-specific spectral parameters like
  ``spectral_energy_range`` which can take custom energy edges as well.
  [`#41 <https://github.com/chaimain/asgardpy/pull/41>`_]

- Add notebooks showing each analysis step separately.
  [`#43 <https://github.com/chaimain/asgardpy/pull/43>`_]


API Changes
-----------

- Restructure pipeline to make it user-friendly and to follow the initiative in
  the Gammapy PR [`#3852 <https://github.com/gammapy/gammapy/pull/3852>`_].
  See Issue [`#24 <https://github.com/chaimain/asgardpy/issues/24>`_] for more
  details.

  Have Asgardpy follow the workflow of the HLI in Gammapy more closely, by
  having a ``Config`` class and an ``Analysis`` class, named as ``AsgardpyConfig``
  and ``AsgardpyAnalysis`` respectively, using ``pydantic``.

  Create a Gammapy ``Registry`` for all the ``AnalysisSteps``.

  Define Base classes for all Config classes and Analysis Steps, and separate
  modules for defining base geometries for DL4 datasets and various dataset
  reduction makers, as ``geom`` and ``reduction`` respectively.

  Rename the Config ``Target_model`` to ``target`` which will contain the target
  source information, required for the high-level analysis.

  Extend support for various I/O options in the ``io`` module

  Distinguish the 1/3 Dataset Config information with the associated Dataset
  type as used in Gammapy. [`#26 <https://github.com/chaimain/asgardpy/pull/26>`_]

- Start compressing the code in various processes to reduce total analysis time.
  [`#36 <https://github.com/chaimain/asgardpy/pull/36>`_]

- Improve the scope to add multiple exclusion regions as a list of
  ``RegionsConfig``, thus removing some hard-coded features.
  [`#45 <https://github.com/chaimain/asgardpy/pull/45>`_]

- Optimize Models assignment with additional inputs of list of dataset names and
  the name of the target source, to read from either the config or the XML file.

  Add a separate notebook, showing the asgardpy processes related with Models.
  [`#46 <https://github.com/chaimain/asgardpy/pull/46>`_]

- Update reading of Model parameters from XML file, by including the
  ``spectrum_type`` information as defined in the original format. This helps
  for Spectral Models like Exponential Cutoff Power Law, Broken Power Law and
  Super-Exponential Cutoff Power Law as used in the 4FGL catalog, where Gammapy
  uses different formulae and parameter names. Resolves a part of the Issue
  [`#52 <https://github.com/chaimain/asgardpy/issues/52>`_].
  [`#53 <https://github.com/chaimain/asgardpy/pull/53>`_]


Bug Fixes
---------

- Fixes ``python_requires`` version in setup.py.
  [`#8 <https://github.com/chaimain/asgardpy/pull/8>`_]

- Try to fix some coding styles to avoid test errors by using isort and
  suggestions from pylint. [`#10 <https://github.com/chaimain/asgardpy/pull/10>`_]

- Update Changelog and fix an earlier commit change.
  [`#17 <https://github.com/chaimain/asgardpy/pull/17>`_]

- Fix adding exclusion regions in 3D dataset and assuming a
  ``CircleAnnulusSkyRegion`` to be the first exclusion region type.
  [`#40 <https://github.com/chaimain/asgardpy/pull/40>`_]

- Fixing assignment of Dataset models to be done in the ``analysis`` module and
  not in each DL4 dataset creation module.

  Check for diffuse background models before enlisting them, and perform any
  additional tasks as required.

  Have the model information of the target source, read from XML file, be the
  first entry in the list of Models.

  Correct the parameter values as defined in Fermi-XML models, by updating the
  units, scaling factors, range of values, and generating a list of Gammapy
  ``Parameter`` objects, to then generate the respective Models object.
  Add links to the Fermi-XML definitions for reference in docstrings.

  Fix the condition on when to use the model information for the target source,
  given in the ``AsgardpyConfig`` file or continue with the information in the
  XML file. [`#42 <https://github.com/chaimain/asgardpy/pull/42>`_]

- Fixing Flux Points Analysis step, to get instrument-specific flux points by
  using ``instrument_spectral_info`` dict object, containing the relevant
  instrument-specific information.

  This information is used to sort the datasets provided for the ``flux-points``
  step, with the respective energy binning and dataset names.
  [`#44 <https://github.com/chaimain/asgardpy/pull/44>`_]

- Cleaning of logging information and updating doc-strings.
  [`#47 <https://github.com/chaimain/asgardpy/pull/47>`_]

- Fix mypy check errors in default values of different variables.
  [`#48 <https://github.com/chaimain/asgardpy/pull/48>`_]

- Fix variable assignment issue from previous PR by using a new Config variable
  ``PathType`` which uses strings of paths and reads them as ``pathlib.Path``
  objects. [`#50 <https://github.com/chaimain/asgardpy/pull/50>`_]

- Clean the pipeline from all outputs, irrelevant comments and reference to any
  private data in config files or notebooks.
  [`#51 <https://github.com/chaimain/asgardpy/pull/51>`_]
