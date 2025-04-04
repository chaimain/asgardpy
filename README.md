Agardpy: Analysis Software for GAmma-Ray Data in Python
=======================================================

[![Build Status](https://github.com/chaimain/asgardpy/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/chaimain/asgardpy/actions?query=branch%3Amain) [![codecov](https://codecov.io/gh/chaimain/asgardpy/branch/main/graph/badge.svg?token=0XEI9W8AKJ)](https://codecov.io/gh/chaimain/asgardpy) [![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/chaimain/asgardpy/badges/quality-score.png?b=main)](https://scrutinizer-ci.com/g/chaimain/asgardpy/?branch=main) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8106369.svg)](https://doi.org/10.5281/zenodo.8106369) ![PyPI](https://img.shields.io/pypi/v/asgardpy?label=pypi%20asgardpy) [![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/7699/badge)](https://bestpractices.coreinfrastructure.org/projects/7699) [![gammapy](https://img.shields.io/badge/powered%20by-gammapy-orange.svg?style=flat)](https://www.gammapy.org/) [![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](https://www.astropy.org/)

'User-friendly' configuration-centred pipeline built over [Gammapy](https://github.com/gammapy/gammapy) to allow for easy simultaneous analysis of various datasets of different formats.
Example: 3D Fermi-LAT (with various source models in the Region of Interest stored in XML file) + 1D energy-dependent directional cuts MAGIC/LST [``PointSkyRegion`` geometry for ``ON`` region] + 1D global directional cut VERITAS [``CircleSkyRegion`` geometry for ``ON`` region].

Follow the documentation at https://asgardpy.readthedocs.io/en/latest/ for the main functionality of this pipeline.
Follow the [Gammapy v1.3](https://docs.gammapy.org/1.3/) documentation for understanding the core Gammapy objects.

Check this [documentation page](https://asgardpy.readthedocs.io/en/latest/need_for_asgardpy.html) for seeing the list of reasons to use Asgardpy over Gammapy v1.3 and this [documentation page](https://asgardpy.readthedocs.io/en/latest/crab_sed.html) for seeing an extended example of the usage of Asgardpy in analyzing multi-instrument data of Crab Nebula.

The various Data Levels	used here follow the descriptions suggested by [GADF v0.3](https://gamma-astro-data-formats.readthedocs.io/en/latest/) and [CTAO](https://www.ctao.org/) Data Model.

# NOTE

For requiring support only for [Gammapy v1.1](https://docs.gammapy.org/1.1/), one may follow the latest Hotfix release [v0.4.4](https://github.com/chaimain/asgardpy/tree/v0.4.4) which benefits from the correct usage of getting EBL-deabsorbed data products as included in [v0.5.0](https://github.com/chaimain/asgardpy/tree/v0.5.0). This can be done by using

```bash
git fetch --tags
git switch -c tags/v0.4.4
```

For creating a conda environment, for this Hotfix release, one can use

```bash
conda env create -f environment_0.4.4.yml
```

and in general, for the latest release,

```bash
conda env create -f environment.yml
```

This method was included after v0.5.0, and for earlier (<v0.4.4) releases, one can simply use the gammapy conda environment and install asgardpy on top of it.

# Pipeline development

The pipeline was developed with first testing with Fermi-LAT ([enrico](https://enrico.readthedocs.io/en/latest/) and [fermipy](https://fermipy.readthedocs.io/en/latest/)) files and LST-1 ([cta-lstchain](https://cta-observatory.github.io/cta-lstchain/)) DL3 files (with energy-dependent and global directional cuts) for point-like sources. It also allows for a preliminary analysis of HAWC datasets (stored in Gammapy-readable data).
The pipeline can be further expanded to support more types of DL3 files of gamma-ray instruments.

# Examples with Data
An example of configuration file that can be used with asgardpy can be found at ``asgardpy/config/template.yaml``

For working with some public data to check the pipeline functionality, one should first download the public dataset available with Gammapy as indicated in [Gammapy v1.3 Introduction](https://docs.gammapy.org/1.3/getting-started/index.html) and then run the ``scripts/download_asgardpy_data.sh`` script to add datasets for the full usage of the pipeline.

Examples of usage of Asgardpy is shown in jupyter [notebooks](https://github.com/chaimain/asgardpy/tree/main/notebooks).

# Cite
If you use Asgardpy in a publication, please cite the exact version you used from Zenodo _Cite_ as https://doi.org/10.5281/zenodo.8106369

# Pipeline Template

Pipeline generated based on the template by [python-package-template](https://github.com/allenai/python-package-template) with additional standards being followed -

- [PEP 517](https://peps.python.org/pep-0517)
- [PEP 518](https://peps.python.org/pep-0518/)
- [PyPA specs following PEP 621](https://packaging.python.org/en/latest/specifications/declaring-project-metadata/#declaring-project-metadata)
- [PEP 660](https://peps.python.org/pep-0660/)
