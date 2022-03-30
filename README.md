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
After doing `poetry install`, you can install `cusignal` and `cupy` in order to use the GPU implementation of the deconvolution
with the following in a conda environment:

```sh
conda install -c rapidsai -c nvidia -c conda-forge \
    cusignal=21.08 python=3.8 cudatoolkit=11.0
```

## Usage
You can import all functions directly from `fppanalysis`, such as

```Python
import fppanalysis as fa

bin_centers, hist = fa.get_hist(Data, N)
```
