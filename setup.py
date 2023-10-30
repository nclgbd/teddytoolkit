"""Python `setup.py` for `rtk` package. Adapted from https://github.com/rochacbruno/python-project-template/blob/main/setup.py."""
from setuptools import setup, find_packages
from rtk import __version__


def get_requirements():
    """Reads requirements.txt and returns packages and git repos separately."""
    with open("requirements.txt") as f:
        requirements = f.read().splitlines()
    packages = [r for r in requirements if not r.startswith("git+")]
    # git_repos = [r.replace("git+", "") for r in requirements if r.startswith("git+")]
    return packages  # , git_repos


requirements = get_requirements()
setup(
    name="rtk",
    version=__version__,
    author="teddygu",
    description="General purpose toolkit for working with medical imaging data. Contains experiment management with Mlflow, Hydra, and AzureML.",
    long_description="",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    # requires=requirements,
)
