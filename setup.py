#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

setup_requirements = [
    "pytest-runner>=5.2",
]

test_requirements = [
    "black>=19.10b0",
    "codecov>=2.1.4",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bumpversion>=0.6.0",
    "coverage>=5.1",
    "ipython>=7.15.0",
    "m2r>=0.2.1",
    "pytest-runner>=5.2",
    "Sphinx>=2.0.0b1,<3",
    "sphinx_rtd_theme>=0.4.3",
    "tox>=3.15.2",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

step_workflow_requirements = [
    "bokeh>=2.0.2",
    "dask[bag]>=2.18.1",
    "dask_jobqueue>=0.7.0",
    "datastep>=0.1.9",
    "distributed>=2.18.0",
    "docutils",
    "fire",
    "prefect",
    "python-dateutil",
    "aics_dask_utils",
]

requirements = [
    *step_workflow_requirements,
    # project requires
    "numpy",
    "pandas",
    "Pillow",
    "matplotlib",
    "seaborn",
    "tqdm",
    "scipy",
    "scikit-image",
    "aicsimageio",
    "imgkit==1.0.2",
    "xvfbwrapper==0.2.9",
    "aicsshparam>=0.1.1",
    "aicscytoparam>=0.1.3",
    "vtk==9.0.1",
    "quilt3",
    "ffmpeg",
    "jupyterlab", 
    "ipywidgets"
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "all": [
        *requirements,
        *dev_requirements,
    ],
}

setup(
    author="Allen Institute for Cell Science",
    author_email="ritvik.vasan@alleninstitute.org",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Analysis of data produced by cvapipe for the variance paper",
    entry_points={"console_scripts": ["cvapipe_analysis=cvapipe_analysis.bin.cli:cli"]},
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="cvapipe_analysis",
    name="cvapipe_analysis",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.7",
    setup_requires=setup_requirements,
    test_suite="cvapipe_analysis/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/AllenCellModeling/cvapipe_analysis",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.1.0",
    zip_safe=False,
)
