asgardpy package
================

Main structure
--------------

The package is structured in 2 ways -

1. Creating the AnalysisConfig based on several Config components

2. Generating AsgardpyAnalysis based on several Analysis Step components

Analysis Steps
--------------

The configuration-based pipeline separates the Gammapy-based High-Level Analysis into serialized intermediate steps.
The steps are:

1. datasets-3d

2. datasets-1d

3. fit

4. flux-points

5. light-curve

The main purpose of this pipeline is accomplished before the "fit" step, which is to compile a Gammapy Datasets object
containing multiple types of datasets from multiple gamma-ray astronomical instruments, update them with appropriate
Gammapy Models object. These are then run with the standard Gammapy high-level analysis functions to get the DL5 products.


DL3 Data component
------------------

The "DL3 level" data files for any instrument is read by providing the path location and a search glob pattern in the Config file. These are read
by the IO module's DL3Files class.

The main modules dealing with the 2 types of data being read are -

1. 3D Dataset - asgardpy/data/dataset_3d

2. 1D Dataset - asgardpy/data/dataset_1d

They each build their Config components using classes defined with,

1. a base in asgardpy/data/base,

2. from distinct modules -

2.1. Base Geometry - asgardpy/data/geom

2.2 Dataset Reduction - asgardpy/data/reduction

3. and from their own respective modules

The processing of Dataset creation is performed by the classes Dataset3DGeneration and Dataset1DGeneration


High-level Analysis
-------------------

The various Config components and Analysis steps for the high-level analysis can be found in the module asgardpy/data/dl4
