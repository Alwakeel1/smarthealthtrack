"""
Set up script for SmartHealth-Track library.
"""

from setuptools import setup, find_packages

setup(
    name="smarthealthtrack",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
    ],
    package_data={
        "smarthealthtrack": ["data/*.csv"],
    },
    description="Synthetic datasets for infectious disease monitoring research",
    author="Mohammed M. Alwakeel",
    author_email="alwakeel@ut.edu.sa",
    url="https://github.com/yourusername/smarthealthtrack",
)
