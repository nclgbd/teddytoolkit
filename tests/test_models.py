"""
Tests for the `rtk.config` module.
"""
# torch
import pytest
import torch
from torch import nn

# monai
from generative.inferers import DiffusionInferer
from generative.networks.schedulers import DDPMScheduler

# rtk
from rtk import models
from rtk.config import Configuration, ModelConfiguration, DiffusionModelConfiguration


class TestModels:
    @pytest.fixture
    def model_cfg(self, test_cfg: Configuration) -> ModelConfiguration:
        """Fixture for the model configuration."""
        return test_cfg.models

    @pytest.fixture
    def diffusion_model_cfg(
        self, test_cfg: Configuration
    ) -> DiffusionModelConfiguration:
        """Fixture for the model configuration."""
        return test_cfg.models

    def test_instantiate_model(self, model_cfg: ModelConfiguration):
        """Test the `rtk.models.instantiate_model` function."""
        model = models.instantiate_model(model_cfg, device=torch.device("cpu"))
        assert isinstance(model, nn.Module)

    def test_instantiate_criterion(self, model_cfg: ModelConfiguration):
        """Test the `rtk.models.instantiate_criterion` function."""
        criterion = models.instantiate_criterion(model_cfg)
        assert isinstance(criterion, nn.Module)

    def test_instantiate_optimizer(self, model_cfg: ModelConfiguration):
        """Test the `rtk.models.instantiate_optimizer` function."""
        model = models.instantiate_model(model_cfg, device=torch.device("cpu"))
        optimizer = models.instantiate_optimizer(model_cfg, model=model)
        assert isinstance(optimizer, torch.optim.Optimizer)

    @pytest.mark.diffusion
    def test_instantiate_diffusion_scheduler(
        self, diffusion_model_cfg: DiffusionModelConfiguration
    ):
        """Test the `rtk.models.instantiate_diffusion_scheduler` function."""
        scheduler: DDPMScheduler = models.instantiate_diffusion_scheduler(
            diffusion_model_cfg
        )
        assert isinstance(scheduler, DDPMScheduler)

    @pytest.mark.diffusion
    def test_instantiate_diffusion_inferer(
        self, diffusion_model_cfg: DiffusionModelConfiguration
    ):
        """Test the `rtk.models.instantiate_diffusion_inferer` function."""
        scheduler: DDPMScheduler = models.instantiate_diffusion_scheduler(
            diffusion_model_cfg
        )
        inferer: DiffusionInferer = models.instantiate_diffusion_inferer(
            model_cfg=diffusion_model_cfg, scheduler=scheduler
        )
        assert isinstance(inferer, DiffusionInferer)
