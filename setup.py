#!/usr/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='dmax',
    version='0.0.1',
    description='Implementation of the the Dmax estimator with latent Optimal Transport',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Theo J. Adrai',
    author_email='tjtadrai@gmail.com',
    url='https://github.com/theoad/dmax',
    install_requires=requirements,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)