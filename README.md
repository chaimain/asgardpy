# asgardpy [![Build Status](https://github.com/chaimain/asgardpy/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/chaimain/asgardpy/actions?query=branch%3Amain) [![gammapy](https://img.shields.io/badge/powered%20by-gammapy-orange.svg?style=flat)](https://www.gammapy.org/) [![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](https://www.astropy.org/)
## Analysis Software for GAmma-Ray Data in Python

'User-friendly' configuration-centred pipeline built over [Gammapy](https://github.com/gammapy/gammapy) to allow for easy simultaneous analysis of various datasets of different formats.
Example: 3D Fermi-LAT (with various source models in the Region of Interest stored in XML file) + 1D energy-dependent selection cut MAGIC/LST [PointSkyRegion geometry for ON region] + 1D fixed-cut VERITAS [CircleSkyRegion geometry for ON region].

Follow the documentation at https://asgardpy.readthedocs.io/en/latest/ for the main functionality of this pipeline.
Follow the [Gammapy v1.1](https://docs.gammapy.org/1.1/) documentation for understanding the core Gammapy objects.

The various Data Levels	used here follow the descriptions suggested by [GADF v0.3](https://gamma-astro-data-formats.readthedocs.io/en/latest/) and CTAO Data Model

# Pipeline development

The pipeline was developed with first testing with Fermi-LAT ([enrico](https://enrico.readthedocs.io/en/latest/) and [fermipy](https://fermipy.readthedocs.io/en/latest/)) files and LST-1 ([cta-lstchain](https://cta-observatory.github.io/cta-lstchain/)) DL3 files (with energy-dependent and global selection cuts) for point-like sources.
The pipeline can be further expanded to support more types of DL3 files of gamma-ray instruments.

An example of configuration file that can be used with asgardpy can be found at ``asgardpy/config/template.yaml``
Examples of usage of asgardpy is shown in jupyter notebooks in ``notebooks/`` but as there are no public test data included with the pipeline yet, the results are empty.

# Pipeline Template

Pipeline generated based on the template by [python-package-template](https://github.com/allenai/python-package-template).
