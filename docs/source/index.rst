.. asgardpy documentation master file, created by
   sphinx-quickstart on Tue Sep 21 08:07:48 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**asgardpy package**
====================

Gammapy-based package to support high-level analysis for multi-instruments joint datasets.


Main structure
--------------

The package is structured in 2 ways -

1. Creating the AnalysisConfig based on several Config components - :doc:`config`

2. Generating AsgardpyAnalysis based on several Analysis Step components - :doc:`analysis`

Analysis Steps
--------------

The configuration-based pipeline separates the Gammapy-based High-Level Analysis into serialized intermediate steps.
Check :doc:`data_base` for more details.
The steps are:

1. datasets-3d

2. datasets-1d

3. fit

4. flux-points

The main purpose of this pipeline is accomplished before the "fit" step, which is to compile a Gammapy Datasets object
containing multiple types of datasets from multiple gamma-ray astronomical instruments, update them with appropriate
Gammapy Models object. These are then run with the standard Gammapy high-level analysis functions to get the DL5 products.


DL3 Data component
------------------

The "DL3 level" data files for any instrument is read by providing the path location and a search glob pattern in the Config file. These are read
by the IO module's DL3Files class.

The main modules dealing with the 2 types of data being read are -

1. 3D Dataset - :doc:`data_3d`

2. 1D Dataset - :doc:`data_1d`

They each build their Config components using classes defined with,

1. a base in asgardpy/data/base,

2. from distinct modules -

2.1. Base Geometry - asgardpy/data/geom

2.2 Dataset Reduction - asgardpy/data/reduction

3. and from their own respective modules

The processing of Dataset creation is performed by the classes Dataset3DGeneration and Dataset1DGeneration

Models
------

The :doc:`data_target` contains separate functions and classes for handling various Models objects.

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
   config
   data_base
   data_3d
   data_1d
   data_target
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
