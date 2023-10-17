"""Python `setup.py` for `rtk` package. Adapted from https://github.com/rochacbruno/python-project-template/blob/main/setup.py."""
from setuptools import setup, find_packages
from rtk import __version__

setup(
    name="researchtoolkit",
    version=__version__,
    author="nclgbd",
    description="Toolkit for working with medical imaging data.",
    long_description="",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    requires=[
        "azureml-core",
        "azureml-dataset-runtime",
        "einops",
        "hydra-colorlog",
        "hydra-core",
        "itk",
        "matplotlib",
        "mlflow",
        "monai",
        "nibabel",
        "numpy",
        "omegaconf",
        "pandas",
        "pydicom",
        "pytorch-ignite",
        "rich",
        "scikit-image",
        "scikit-learn",
        "torch",
        "torchvision",
        "tqdm",
        "xlrd",
    ],
)
