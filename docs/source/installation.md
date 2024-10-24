Installation
============

**asgardpy** supports Python >= 3.11.

## Installing with `pip`

**asgardpy** is available [on PyPI](https://pypi.org/project/asgardpy/). For the latest version just run

```bash
pip install asgardpy
```

and for specific versions, run

```bash
pip install asgardpy==VERSION
```

For example, using the Hotfix release of v0.4.4 with extended support for Gammapy v1.1, run

```bash
pip install asgardpy==0.4.4
```

## Installing from source

To install **asgardpy** from source, first clone [the repository](https://github.com/chaimain/asgardpy):

```bash
git clone https://github.com/chaimain/asgardpy.git
cd asgardpy
```

Then for users, run

```bash
pip install -e .
```

and for developers, run

```bash
pip install -e .[dev]
```

## Creating conda environment

In general, for the latest version, one can use

```bash
conda env create -f environment.yml
```

and for the Hotfix release,

```bash
conda env create -f environment_0.4.4.yml
```

This method was included after v0.5.0, and for earlier (<v0.4.4) releases, one can simply use the gammapy conda environment and install asgardpy on top of it.

## Downloading Test data

To download the primary source of Test data, ie, the public data available with gammapy,
follow the instructions as mentioned in [Gammapy v1.3 Introduction](https://docs.gammapy.org/1.3/getting-started/index.html), by running the following,

```bash
gammapy download datasets
export GAMMAPY_DATA=$PWD/gammapy-datasets/1.3/
```

Next, to add the extra test data for asgardpy tests, run the following,

```bash
./scripts/download_asgardpy_data.sh
```

This adds the additional datasets in the same location as ``GAMMAPY_DATA``.
