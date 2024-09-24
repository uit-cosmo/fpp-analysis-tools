# fpp-analysis-tools

Collection of tools designed to analyse time series of intermittent fluctuations.

## Installation

The package ~~is published to [PyPI] and~~ can be installed with

```sh
pip install git+https://github.com/uit-cosmo/fpp-analysis-tools
```

If you want to contribute to the project you must first clone the repo to your local
machine, then install the project using [poetry]:

```sh
git clone git@github.com:uit-cosmo/fpp-analysis-tools.git
cd fpp-analysis-tools
poetry install
```

If you plan to use the GPUs, specifically useful for the deconvolution, (local)
installation using both [pixi] and [conda] is supported (the conda environment file is
exported by pixi):

```sh
git clone git@github.com:uit-cosmo/fpp-analysis-tools.git
cd fpp-analysis-tools
# pixi
pixi install
# conda
conda env create --name name-of-my-env --file environment.yml
```

## Usage

You can import all functions directly from `fppanalysis`, such as

```python
import fppanalysis as fa

bin_centers, hist = fa.get_hist(Data, N)
```

[conda]: https://docs.conda.io/en/latest/index.html
[poetry]: https://python-poetry.org/
[pixi]: https://pixi.sh/latest/
[pypi]: https://pypi.org/
