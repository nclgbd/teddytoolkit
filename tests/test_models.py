"""
Tests for the `rtk.config` module.
"""

from collections import Counter
import pytest

# torch
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader

# monai
from generative.inferers import DiffusionInferer
from generative.networks.schedulers import DDPMScheduler

# rtk
from rtk import models
from rtk.config import Configuration, ModelConfiguration, DiffusionModelConfiguration
from rtk.datasets import (
    _LABEL_KEYNAME,
    create_transforms,
    instantiate_image_dataset,
    instantiate_train_val_test_datasets,
)
from rtk.utils import hydra_instantiate


class TestModels:
    @pytest.fixture
    def samples_per_class(self, test_cfg: Configuration):
        """Fixture for the train loader."""
        job_cfg = test_cfg.job
        use_transforms = job_cfg.use_transforms
        transform = create_transforms(test_cfg, use_transforms=use_transforms)
        val_transform = create_transforms(test_cfg, use_transforms=False)
        _datasets = instantiate_image_dataset(cfg=test_cfg, transform=transform)
        _train_dataset = _datasets[0]
        train_val_test_split_dict = instantiate_train_val_test_datasets(
            cfg=test_cfg,
            dataset=_train_dataset,
            train_transforms=transform,
            eval_transforms=val_transform,
        )
        train_dataset = train_val_test_split_dict["train"]
        values_counts = list(Counter(vars(train_dataset)[_LABEL_KEYNAME]).values())
        return values_counts

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

    def test_instantiate_model(self, test_cfg: Configuration):
        """Test the `rtk.models.instantiate_model` function."""
        model = models.instantiate_model(test_cfg, device=0)
        assert isinstance(model, nn.Module)

    def test_instantiate_criterion(
        self, test_cfg: Configuration, samples_per_class: list
    ):
        """Test the `rtk.models.instantiate_criterion` function."""

        if test_cfg.models.criterion._target_.split(".")[0] == "balanced_loss":
            criterion = models.instantiate_criterion(
                test_cfg,
                device=torch.device("cpu"),
                samples_per_class=samples_per_class,
            )
            assert isinstance(criterion, nn.Module)
        else:
            criterion = models.instantiate_criterion(
                test_cfg, device=torch.device("cpu")
            )
            assert isinstance(criterion, nn.Module)

    def test_instantiate_optimizer(self, test_cfg: Configuration):
        """Test the `rtk.models.instantiate_optimizer` function."""
        model = models.instantiate_model(test_cfg, device=torch.device("cpu"))
        optimizer = models.instantiate_optimizer(test_cfg, model=model)
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
