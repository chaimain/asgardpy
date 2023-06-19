.. asgardpy documentation master file, created by
   sphinx-quickstart on Tue Sep 21 08:07:48 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**asgardpy pipeline**
=====================

Gammapy-based pipeline to support high-level analysis for multi-instruments joint datasets.
Follow the `Gammapy v1.1 <https://docs.gammapy.org/1.1/>`_ documentation for understanding the core Gammapy objects.

The various Data Levels used here follow the descriptions suggested by `GADF v0.3 <https://gamma-astro-data-formats.readthedocs.io/en/latest/>`_ and CTAO Data Model.

Main structure
--------------

The package is structured in 2 ways -

#. Creating the AnalysisConfig based on several Config components - :doc:`config`

#. Generating AsgardpyAnalysis based on several Analysis Step components - :doc:`analysis`

Analysis Steps
--------------

The configuration-based pipeline separates the Gammapy-based High-Level Analysis into serialized intermediate steps.
Check :doc:`base_base` for more details.
The steps are:

#. datasets-3d :class:`asgardpy.data.dataset_3d.Datasets3DAnalysisStep`

#. datasets-1d :class:`asgardpy.data.dataset_1d.Datasets1DAnalysisStep`

#. fit :class:`asgardpy.data.dl4.FitAnalysisStep`

#. flux-points :class:`asgardpy.data.dl4.FluxPointsAnalysisStep`

The main purpose of this pipeline is accomplished before the "fit" step, which is to compile a Gammapy Datasets object
containing multiple types of datasets from multiple gamma-ray astronomical instruments, update them with appropriate
Gammapy Models object. These are then run with the standard Gammapy high-level analysis functions to get the DL5 products.


.. image:: _static/asgardpy_workflow.png
    :width: 600px
    :align: center


DL3 Data component
------------------

The "DL3 level" data files for any instrument is read by providing the path location and a search glob pattern in the Config file. These are read
by the :class:`asgardpy.io.io.DL3Files`.

The main modules dealing with the 2 types of data being read are -

#. 3D Dataset :doc:`data_3d`

#. 1D Dataset :doc:`data_1d`

They each build their Config components using classes defined with,

#. a base in :class:`asgardpy.base.base`,

#. from distinct modules -

   #. Base Geometry :doc:`base_geom`

   #. Dataset Reduction :doc:`base_reduction`

#. and from their own respective modules

The processing of Dataset creation is performed by :class:`asgardpy.data.dataset_3d.Dataset3DGeneration` and :class:`asgardpy.data.dataset_1d.Dataset1DGeneration`

Models
------

The :doc:`data_target_b` contains various classes for various Models objects and :doc:`data_target_f` contains various functions for handling them.

High-level Analysis
-------------------

The various Config components and Analysis steps for the high-level analysis can be found in :doc:`data_dl4`.

Getting started
---------------

.. toctree::
   :maxdepth: 2

   installation
   CHANGELOG


.. toctree::
  :hidden:
  :caption: Development

  License <https://github.com/chaimain/asgardpy/blob/main/LICENSE>
  CONTRIBUTING
  GitHub Repository <https://github.com/chaimain/asgardpy>



Contents
--------

.. toctree::
   :maxdepth: 2

   analysis
   base_base
   base_geom
   base_reduction
   config
   data_3d
   data_1d
   data_target_b
   data_target_f
   data_dl4
   io


Team
----

**asgardpy** is developed and maintained by `Chaitanya Priyadarshi <https://github.com/chaimain>`_.
To learn more about who specifically contributed to this codebase, see
`our contributors <https://github.com/chaimain/asgardpy/graphs/contributors>`_ page.

License
-------

**asgardpy** is licensed under `Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
A full copy of the license can be found on `GitHub <https://github.com/chaimain/asgardpy/blob/main/LICENSE>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`


Dependencies
------------

* `astropy <https://www.astropy.org>`_ managing physical units and astronomical distances;

* `gammapy <https://gammapy.org/>`_ for main high-level analysis
