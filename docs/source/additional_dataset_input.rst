Additional Support of Datasets input
====================================


:ref:`dataset-intro` provides the general overview of the inputs used in asgardpy
for the data reduction of DL3 data to DL4 data, following the Gammapy documentation on
`Data Reduction (DL3 to DL4) <https://docs.gammapy.org/1.1/user-guide/makers/index.html>`_


Asgardpy can also read from already reduced DL4 datasets, like the examples from gammapy-data,
using the module :class:`~asgardpy.io.io_dl4`. This is done by using
:class:`~asgardpy.data.dataset_3d.Dataset3DBaseConfig.input_dl4` ``= True``
or :class:`~asgardpy.data.dataset_1d.Dataset1DBaseConfig.input_dl4` ``= True`` and filling
:class:`~asgardpy.data.dataset_3d.Dataset3DBaseConfig.dl4_dataset_info` and
:class:`~asgardpy.data.dataset_1d.Dataset1DBaseConfig.dl4_dataset_info` respectively.

This can be done for an additional compilation of DL4 datasets to run the joint
likelihood analysis using Gammapy. Tests are included for checking an example
of this support.
