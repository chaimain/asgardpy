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
pip install asgardpy=VERSION
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

## Downloading Test data

To download the primary source of Test data, ie, the public data available with gammapy,
follow the instructions as mentioned in [Gammapy v1.2 Introduction](https://docs.gammapy.org/1.2/getting-started/index.html), by running the following,

```bash
gammapy download datasets
export GAMMAPY_DATA=$PWD/gammapy-datasets/1.2
```

Next, to add the extra test data for asgardpy tests, run the following,

```bash
./scripts/download_asgardpy_data.sh
```
