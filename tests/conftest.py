import os
import pytest

from rtk.config import set_hydra_configuration, ImageClassificationConfiguration


@pytest.fixture
def test_config_name():
    """Fixture for the test configuration name."""
    return "tests"


@pytest.fixture
def test_config_dir():
    """Fixture for the test configuration directory."""
    return os.path.abspath("configs")


@pytest.fixture
def test_cfg(test_config_name: str, test_config_dir: os.PathLike):
    """Fixture for the rtk configuration."""
    return set_hydra_configuration(
        config_name=test_config_name,
        init_method_kwargs={"config_dir": test_config_dir},
        BaseConfigurationInstance=ImageClassificationConfiguration,
    )
