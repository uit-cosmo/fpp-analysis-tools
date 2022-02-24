# fpp-analysis-tools
Collection of tools designed to analyse time series of intermittent fluctuations.

## Installation
The package is published to PyPI and can be installed with
```sh
pip install fpp-analysis-tools
```

If you want the development version you must first clone the repo to your local machine,
then install the project in development mode:

```sh
git clone git@github.com:uit-cosmo/fpp-analysis-tools.git
cd fpp-analysis-tools
pip install -e .
```

## Usage
You can import all functions directly from `analysistools`, such as

```Python
import analysistools as at

bin_centers, hist = at.get_hist(Data, N)
```