"""
Setup script for Spatial-Omics GFM.

This setup script is provided for compatibility with older build systems.
The main configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages

setup(
    name="spatial-omics-gfm",
    use_scm_version=True,
    packages=find_packages(),
    python_requires=">=3.9",
)