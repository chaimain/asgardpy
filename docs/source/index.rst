.. asgardpy documentation master file, created by
   sphinx-quickstart on Tue Sep 21 08:07:48 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**asgardpy package**
====================

Gammapy-based package to support high-level analysis for multi-instruments joint datasets.
Follow the `Gammapy v1.1 <https://docs.gammapy.org/1.1/>`_ documentation for understanding the core Gammapy objects.

The various Data Levels used here follow the descriptions suggested by
`GADF v0.3 <https://gamma-astro-data-formats.readthedocs.io/en/latest/>`_ and CTAO Data Model.

The package was developed with first testing with Fermi-LAT (`enrico <https://enrico.readthedocs.io/en/latest/>`_ and
`fermipy <https://fermipy.readthedocs.io/en/latest/>`_) files and LST-1 (`cta-lstchain <https://cta-observatory.github.io/cta-lstchain/>`_)
DL3 files (with energy-dependent and global selection cuts) for point-like sources.
The package can be further expanded to support more types of DL3 files of gamma-ray instruments.

GitHub Repository: https://github.com/chaimain/asgardpy

.. _introduction:

.. toctree::
   :maxdepth: 2
   :caption: Introduction
   :name: _introduction

   overview
   installation
   CHANGELOG

.. _api:

.. toctree::
   :maxdepth: 1
   :caption: API Documentation
   :name: _api

   _api_docs/index

.. toctree::
  :hidden:
  :caption: Development

  License <https://github.com/chaimain/asgardpy/blob/main/LICENSE>
  CONTRIBUTING
  GitHub Repository <https://github.com/chaimain/asgardpy>


Team
----

**asgardpy** is developed and maintained by `Chaitanya Priyadarshi <https://github.com/chaimain>`_.
To learn more about who specifically contributed to this codebase, see
`our contributors <https://github.com/chaimain/asgardpy/graphs/contributors>`_ page.


Cite
----

If you use Asgardpy in a publication, please cite the exact version you used from Zenodo *Cite as* https://doi.org/10.5281/zenodo.8106369


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
