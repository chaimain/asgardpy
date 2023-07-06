Overview of asgardpy
====================

Main structure
--------------

The package is structured in 2 ways -

#. Creating the AnalysisConfig based on several Config components - :doc:`_api_docs/config`

#. Generating AsgardpyAnalysis based on several Analysis Step components - :doc:`_api_docs/analysis`

Analysis Steps
--------------

The configuration-based package separates the Gammapy-based High-Level Analysis into serialized intermediate steps.
Check :doc:`_api_docs/base_base` for more details.
The steps are:

#. datasets-3d :class:`asgardpy.data.dataset_3d.Datasets3DAnalysisStep`

#. datasets-1d :class:`asgardpy.data.dataset_1d.Datasets1DAnalysisStep`

#. fit :class:`asgardpy.data.dl4.FitAnalysisStep`

#. flux-points :class:`asgardpy.data.dl4.FluxPointsAnalysisStep`

The main purpose of this package is accomplished before the "fit" step, which is to compile a Gammapy Datasets object
containing multiple types of datasets from multiple gamma-ray astronomical instruments, update them with appropriate
Gammapy Models object. These are then run with the standard Gammapy high-level analysis functions to get the DL5 products.


.. image:: ./_static/asgardpy_workflow.png
    :width: 600px
    :align: center


DL3 Data component
------------------

The "DL3 level" data files for any instrument is read by providing the path location and a search glob pattern in the Config file. These are read
by the :class:`asgardpy.io.io.DL3Files`.

The main modules dealing with the 2 types of data being read are -

#. 3D Dataset :doc:`_api_docs/data_3d`

#. 1D Dataset :doc:`_api_docs/data_1d`

They each build their Config components using classes defined with,

#. a base in :class:`asgardpy.base.base`,

#. from distinct modules -

   #. Base Geometry :doc:`_api_docs/base_geom`

   #. Dataset Reduction :doc:`_api_docs/base_reduction`

#. and from their own respective modules

The processing of Dataset creation is performed by :class:`asgardpy.data.dataset_3d.Dataset3DGeneration` and :class:`asgardpy.data.dataset_1d.Dataset1DGeneration`

For the DL3 files with energy-dependent directional (RAD_MAX) cuts, the ON region is defined by a `PointSkyRegion` whereas for files with global cuts, the ON region is defined by a `CircleSkyRegion`.
The usage of either of these can be generalized by providing the source sky position with a value of radius as 0 deg (by default) or a non-zero angular radius, respectively.

Following `Gammapy v1.1 <https://docs.gammapy.org/1.1/>`_ we have the usage of parallel processing for DL4 Dataset creation, Flux Points Estimation and also Light Curve Estimation.
For the first two processes, here we have the parameters of `n_jobs` and `parallel_backend` defined in `asgardpy.config.generator.GeneralConfig` as can be seen in :class:`asgardpy.config.AsgardpyConfig`.

Models
------

The :doc:`_api_docs/data_target_b` contains various classes for various Models objects and :doc:`_api_docs/data_target_f` contains various functions for handling them.

High-level Analysis
-------------------

The various Config components and Analysis steps for the high-level analysis can be found in :doc:`_api_docs/data_dl4`.
