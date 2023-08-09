"""
Tests for the `ttk.config` module.
"""
# torch
import torch
from torch import nn

# monai
from generative.inferers import DiffusionInferer
from generative.networks.schedulers import DDPMScheduler

# ttk
from ttk import models
from ttk.config import Configuration, ModelConfiguration, DiffusionModelConfiguration


class TestModels:
    def test_instantiate_model(self, test_cfg: Configuration):
        """Test the `ttk.models.instantiate_model` function."""
        model = models.instantiate_model(test_cfg.models, device=torch.device("cpu"))
        assert isinstance(model, nn.Module)

    def test_instantiate_criterion(self, test_cfg: Configuration):
        """Test the `ttk.models.instantiate_criterion` function."""
        criterion = models.instantiate_criterion(test_cfg.models)
        assert isinstance(criterion, nn.Module)

    def test_instantiate_optimizer(self, test_cfg: Configuration):
        """Test the `ttk.models.instantiate_optimizer` function."""
        model = models.instantiate_model(test_cfg.models, device=torch.device("cpu"))
        optimizer = models.instantiate_optimizer(test_cfg.models, model)
        assert isinstance(optimizer, torch.optim.Optimizer)

    def test_instantiate_diffusion_scheduler(self, test_cfg: Configuration):
        """Test the `ttk.models.instantiate_diffusion_scheduler` function."""
        scheduler: DDPMScheduler = models.instantiate_diffusion_scheduler(
            test_cfg.models
        )
        assert isinstance(scheduler, DDPMScheduler)

    def test_instantiate_diffusion_inferer(self, test_cfg: Configuration):
        """Test the `ttk.models.instantiate_diffusion_inferer` function."""
        scheduler: DDPMScheduler = models.instantiate_diffusion_scheduler(
            test_cfg.models
        )
        inferer: DiffusionInferer = models.instantiate_diffusion_inferer(
            model_cfg=test_cfg.models, scheduler=scheduler
        )
        assert isinstance(inferer, DiffusionInferer)
