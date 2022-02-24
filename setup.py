import os
from setuptools import setup

name = "analysistools"

with open("README.md") as f:
    long_description = f.read()

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name=name,
    description="Analysis tools for time series",
    url="https://github.com/uit-cosmo/fpp-analysis-tools",
    author="Gregor Decristoforo",
    author_email="gregor.decristoforo@uit.no",
    license="MIT",
    version="0.1.0",
    packages=["analysistools"],
    python_requires=">=3.9",
    install_requires=[
        "numpy >= 1.22.2",
        "scipy >= 1.8.0",
        "tqdm >= 4.62.3",
        "PyWavelets >= 1.1.1",
    ],
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
    ],
    zip_safe=False,
)
