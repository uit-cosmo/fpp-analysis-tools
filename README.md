# fpp-analysis-tools

Collection of tools designed to analyse time series of intermittent fluctuations.

## Installation

The package is published to PyPI and can be installed with

```sh
pip install fppanalysis
```

If you want the development version you must first clone the repo to your local machine,
then install the project in development mode:

```sh
git clone git@github.com:uit-cosmo/fpp-analysis-tools.git
cd fpp-analysis-tools
poetry install
```

If you plan to use the GPUs, specifically for the deconvolution then setup the following conda environment:

```sh
conda create --name my-env
conda activate my-env
conda install -c rapidsai -c nvidia -c conda-forge \
    cusignal=21.08 python=3.9 cudatoolkit=11.0
conda install poetry 
poetry install
```

## Usage

You can import all functions directly from `fppanalysis`, such as

```Python
import fppanalysis as fa

bin_centers, hist = fa.get_hist(Data, N)
```
