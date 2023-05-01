asgardpy.data.dataset_3d module
===============================

Basic overview
--------------

The class :class:`asgardpy.data.dataset_3d.Datasets3DAnalysisStep` gathers information for a list of instruments by using the class :class:`asgardpy.data.dataset_3d.Dataset3DGeneration`.

The main config for this dataset is defined by :class:`asgardpy.data.dataset_3d.Dataset3DConfig` which is simply a collection of basic information as defined in :class:`asgardpy.data.dataset_3d.Dataset3DBaseConfig`. This collection is a combination of :class:`asgardpy.io.io.InputConfig` and :class:`asgardpy.data.dataset_3d.Dataset3DInfoConfig` information.

Classes
-------

.. automodule:: asgardpy.data.dataset_3d
   :members: Dataset3DGeneration, Datasets3DAnalysisStep, Dataset3DBaseConfig, Dataset3DInfoConfig, Dataset3DConfig
   :undoc-members:
   :show-inheritance:
