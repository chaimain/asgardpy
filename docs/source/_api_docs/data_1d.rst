asgardpy.data.dataset_1d module
===============================

Basic overview
--------------

The class :class:`asgardpy.data.dataset_1d.Datasets1DAnalysisStep` gathers information for a list of instruments by using the class :class:`asgardpy.data.dataset_1d.Dataset1DGeneration`.

The main config for this dataset is defined by :class:`asgardpy.data.dataset_1d.Dataset1DConfig` which is simply a collection of basic information as defined in :class:`asgardpy.data.dataset_1d.Dataset1DBaseConfig`. This collection is a combination of :class:`asgardpy.io.io.InputConfig` and :class:`asgardpy.data.dataset_1d.Dataset1DInfoConfig` information.

Classes
-------

.. automodule:: asgardpy.data.dataset_1d
   :members: Dataset1DGeneration, Datasets1DAnalysisStep, Dataset1DBaseConfig, Dataset1DInfoConfig, Dataset1DConfig
   :undoc-members:
   :show-inheritance:
