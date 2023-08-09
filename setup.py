"""Python `setup.py` for `ttk` package. Adapted from https://github.com/rochacbruno/python-project-template/blob/main/setup.py."""
from setuptools import setup, find_packages
from ttk import __version__

setup(
    name="teddytoolkit",
    version=__version__,
    author="nclgbd",
    description="Toolkit for working with medical imaging data.",
    long_description="",
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", ".github"]),
    requires=[
        "azureml-core",
        "azureml-dataset-runtime",
        "hydra-colorlog",
        "hydra-core",
        "ignite",
        "itk",
        "matplotlib",
        "mlflow",
        "monai",
        "nibabel",
        "numpy",
        "omegaconf",
        "pandas",
        "pydicom",
        "rich",
        "scikit-image",
        "scikit-learn",
        "torch",
        "torchvision",
        "tqdm",
        "xlrd",
    ],
)
