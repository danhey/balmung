#!/usr/bin/env python

from setuptools import setup

setup(
    name="balmung",
    license="MIT",
    packages=["balmung"],
    install_requires=['numpy>=1.10','astropy>=1.0','scipy', 'tqdm']
)