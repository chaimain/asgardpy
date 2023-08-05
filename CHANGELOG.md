# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add codespell to dev-requirements. [#104](https://github.com/chaimain/asgardpy/pull/104)

### Fixed

- Improve logging. [#102](https://github.com/chaimain/asgardpy/pull/102)
- Fix estimation of Goodness of Fit statistics. [#106](https://github.com/chaimain/asgardpy/pull/106)
- Resolve the issue of circular imports. [#107](https://github.com/chaimain/asgardpy/pull/107)
- Fix reading of models_file with correct process. [#111](https://github.com/chaimain/asgardpy/pull/111)

### Changed

- Restructure statistics functions and update computation of fit statistics. [#103](https://github.com/chaimain/asgardpy/pull/103)
- Compress and update sphinx docs. [#105](https://github.com/chaimain/asgardpy/pull/105)
- Update python dependency to 3.11 and update docs. [#109](https://github.com/chaimain/asgardpy/pull/109)
- Restructure pipeline to prepare to use public test data. [#110](https://github.com/chaimain/asgardpy/pull/110)

## [v0.3.5](https://github.com/chaimain/asgardpy/releases/tag/v0.3.5) - 2023-07-17

### Fixed

- General clean-up and addition of docstrings to various Configs. [#99](https://github.com/chaimain/asgardpy/pull/99)
- Fix the Changelog to be more descriptive. [#100](https://github.com/chaimain/asgardpy/pull/100)

### Changed

- Update docs with Zenodo DOI. [#95](https://github.com/chaimain/asgardpy/pull/95)
- Restructure pipeline to regroup common functions for GADF-based DL3 files for 1D and 3D dataset. (See Issue [#85](https://github.com/chaimain/asgardpy/issues/85)) [#94](https://github.com/chaimain/asgardpy/pull/94)
- Update documentation and notebooks for the restructured pipeline. [#97](https://github.com/chaimain/asgardpy/pull/97)
- Restructure pipeline for better handling of model association. (See Issue [#85](https://github.com/chaimain/asgardpy/issues/85)) [#98](https://github.com/chaimain/asgardpy/pull/98)
- Update fit-statistics information for the analysis. [#101](https://github.com/chaimain/asgardpy/pull/101)

## [v0.3.4](https://github.com/chaimain/asgardpy/releases/tag/v0.3.4) - 2023-07-02

### Added

- Add script to get most preferred spectral model fit. [#87](https://github.com/chaimain/asgardpy/pull/87)

### Changed

- Combine the various Sky Position configs into a single SkyPositionConfig. [#88](https://github.com/chaimain/asgardpy/pull/88)
- Update documentation for the extended support. [#89](https://github.com/chaimain/asgardpy/pull/89)
- Update with usage of common multiprocessing with Gammapy. [#90](https://github.com/chaimain/asgardpy/pull/90)
- Constrain pydantic and autodoc-pydantic versions. [#92](https://github.com/chaimain/asgardpy/pull/92)

## [v0.3.3](https://github.com/chaimain/asgardpy/releases/tag/v0.3.3) - 2023-06-20

### Added

- Fill more detailed information in the sphinx documentation. [#81](https://github.com/chaimain/asgardpy/pull/81)

### Changed

- Update dependencies and update requirement conditions. [#84](https://github.com/chaimain/asgardpy/pull/84)
- Restructure sphinx documentation to be more compact. [#86](https://github.com/chaimain/asgardpy/pull/86)

### Fixed

- Fix sphinx documentation build issue. [#78](https://github.com/chaimain/asgardpy/pull/78)
- Try to fix sphinx documentation build issue. [#79](https://github.com/chaimain/asgardpy/pull/79)

## [v0.3.2](https://github.com/chaimain/asgardpy/releases/tag/v0.3.2) - 2023-04-28

### Added

- Add the custom spectral models to the Gammapy registry. [#77](https://github.com/chaimain/asgardpy/pull/77)

## [v0.3.1](https://github.com/chaimain/asgardpy/releases/tag/v0.3.1) - 2023-04-28

### Removed

- Remove support of python 3.8. [#76](https://github.com/chaimain/asgardpy/pull/76)

## [v0.3.0](https://github.com/chaimain/asgardpy/releases/tag/v0.3.0) - 2023-04-28

### Changed

- Update documentation with more description. [#73](https://github.com/chaimain/asgardpy/pull/73)
- Update config with the option to perform recursive merging. (See Issue [#71](https://github.com/chaimain/asgardpy/issues/71)) [#72](https://github.com/chaimain/asgardpy/pull/72)
- Restructure a distinct base module. [#75](https://github.com/chaimain/asgardpy/pull/75)

### Fixed

- Fix Safe Mask reduction for 1D Dataset and add another custom Spectral Model. [#74](https://github.com/chaimain/asgardpy/pull/74)

## [v0.2.0](https://github.com/chaimain/asgardpy/releases/tag/v0.2.0) - 2023-04-19

### Added

- Build some custom SpectralModel classes. [#59](https://github.com/chaimain/asgardpy/pull/59)
- Incorporate fermipy files into Dataset3DGeneration. [#61](https://github.com/chaimain/asgardpy/pull/61)
- Add support for common data types for different instruments. (See Issue [#34](https://github.com/chaimain/asgardpy/issues/34)) [#65](https://github.com/chaimain/asgardpy/pull/65)
- Add support for selecting various spectral model parameters in a given Field of View. [#67](https://github.com/chaimain/asgardpy/pull/67)

### Changed

- Update documentation. [#54](https://github.com/chaimain/asgardpy/pull/54)
- Using separate yaml file for Target source. [#57](https://github.com/chaimain/asgardpy/pull/57)
- Using custom EBL models from fits files. [#58](https://github.com/chaimain/asgardpy/pull/58)
- Generalize usage of GeomConfig and Models for analysis with fermipy files. (See Issues [#28](https://github.com/chaimain/asgardpy/issues/28) and [#52](https://github.com/chaimain/asgardpy/issues/52)) [#64](https://github.com/chaimain/asgardpy/pull/64)
- Generalize reading energy axes. [#68](https://github.com/chaimain/asgardpy/pull/68)
- Update notebooks. [#69](https://github.com/chaimain/asgardpy/pull/69)

### Removed

- Remove dependency of hard-coded LAT files structure. [#56](https://github.com/chaimain/asgardpy/pull/56)
- Remove unnecessary features. (See Issue [#60](https://github.com/chaimain/asgardpy/issues/60)) [#62](https://github.com/chaimain/asgardpy/pull/62)
- Remove GTI selections from 3D datasets. [#70](https://github.com/chaimain/asgardpy/pull/70)

## [0.1](https://github.com/chaimain/asgardpy/releases/tag/v0.1) - 2023-02-16

### Added

- Start adding requirements and dependencies. [#6](https://github.com/chaimain/asgardpy/pull/6)
- Start with some IO classes and functions for DL3 and DL4 files. [#7](https://github.com/chaimain/asgardpy/pull/7)
- Start entering Fit and plot functions. [#11](https://github.com/chaimain/asgardpy/pull/11)
- Initial template for the pipeline. [#15](https://github.com/chaimain/asgardpy/pull/15)
- Build pipeline structure. [#16](https://github.com/chaimain/asgardpy/pull/16)
- Add release-drafter in github workflow. [#18](https://github.com/chaimain/asgardpy/pull/18)
- Begin preparations for adding workable scripts. [#19](https://github.com/chaimain/asgardpy/pull/19)
- Work on various Models functions and assignments. (See Issue [#29](https://github.com/chaimain/asgardpy/issues/29)) [#31](https://github.com/chaimain/asgardpy/pull/31)
- Adding plotting functions and flux-points step. [#32](https://github.com/chaimain/asgardpy/pull/32)
- Start using time intervals and light-curve analysis step. (See Issue [#30](https://github.com/chaimain/asgardpy/issues/30)) [#35](https://github.com/chaimain/asgardpy/pull/35)
- Start adding example notebooks. [#37](https://github.com/chaimain/asgardpy/pull/37)
- Addition of instrument-specific spectral parameters. [#41](https://github.com/chaimain/asgardpy/pull/41)
- Add more notebooks testing each analysis step. [#43](https://github.com/chaimain/asgardpy/pull/43)

### Fixed

- Trying to fix python_requires version value. [#8](https://github.com/chaimain/asgardpy/pull/8)
- Try to fix some coding styles to avoid test errors. [#10](https://github.com/chaimain/asgardpy/pull/10)
- Update Changelog and fix an earlier commit change. [#17](https://github.com/chaimain/asgardpy/pull/17)
- Fix adding exclusion regions in 3D dataset. [#40](https://github.com/chaimain/asgardpy/pull/40)
- Fixing assignment of Dataset models. [#42](https://github.com/chaimain/asgardpy/pull/42)
- Fixing Flux Points Analysis step, to get instrument-specific flux points. [#44](https://github.com/chaimain/asgardpy/pull/44)
- Cleaning of logging information and updating doc-strings. [#47](https://github.com/chaimain/asgardpy/pull/47)
- Fix mypy check errors in default variable assignments. [#48](https://github.com/chaimain/asgardpy/pull/48)
- Fix variable assignment issue from previous PR. [#50](https://github.com/chaimain/asgardpy/pull/50)
- Clean pipeline. [#51](https://github.com/chaimain/asgardpy/pull/51)

### Changed

- Restructuring pipeline. (See Issue [#24](https://github.com/chaimain/asgardpy/issues/24)) [#26](https://github.com/chaimain/asgardpy/pull/26)
- Start compressing the pipeline to reduce analysis time. [#36](https://github.com/chaimain/asgardpy/pull/36)
- Improve the scope to add multiple exclusion regions. [#45](https://github.com/chaimain/asgardpy/pull/45)
- Optimize Models section of the pipeline. [#46](https://github.com/chaimain/asgardpy/pull/46)
- Updating Model parameters to read from XML file. (See Issue [#52](https://github.com/chaimain/asgardpy/issues/52)) [#53](https://github.com/chaimain/asgardpy/pull/53)

[Unreleased]: https://github.com/chaimain/keep-a-changelog/compare/v0.1...HEAD
[0.1]: https://github.com/chaimain/asgardpy/releases/tag/0.1
