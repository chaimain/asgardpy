Reasons for using Asgardpy over Gammapy v1.3
============================================

Main reasons
------------

The simple main reasons are the following -

#. For `3D + 1D joint analysis <https://docs.gammapy.org/1.3/tutorials/analysis-3d/analysis_mwl.html>`_, starting from DL3 level data

#. Using `HLI <https://docs.gammapy.org/1.3/user-guide/hli.html>`_ for 1D with energy-dependent RAD_MAX values

#. Additional supporting features while using `HLI <https://docs.gammapy.org/1.3/user-guide/hli.html>`_

Unique reasons
--------------

Some of the unique features in asgardpy are -

#. While creating a multi-instrument list of DL4 datasets, letting the central spatial coordinate in the ``geom`` objects, be commonly shared

#. Being able to read Fermi files as produced by `enrico <https://enrico.readthedocs.io/en/latest/>`_

#. Useful scripts and functions to help with the spectral analyses.

Reasons that may be added in future Gammapy releases
----------------------------------------------------

The following features may become redundant in asgardpy, after `Gammapy 2.0 <https://github.com/gammapy/gammapy/milestone/31>`_ -

#. Being able to read Fermi files as produced by `fermipy <https://fermipy.readthedocs.io/en/latest/>`_

#. Easily reading existing Fermi XML files into Gammapy ``Models`` objects

#. Using the High Level Interface for 1D dataset with energy-dependent ``RAD_MAX`` cuts (see `Workflow module <https://github.com/gammapy/gammapy/blob/main/gammapy/workflow/>`_)

#. Having intermediate analysis steps, distinct for DL3 -> DL4 -> DL5 (see `Workflow module <https://github.com/gammapy/gammapy/blob/main/gammapy/workflow/>`_)

#. Providing Goodness of Fit estimation for 3D + 1D datasets

#. Using multiple time interval selection for a given list of observations
