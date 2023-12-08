"""
Tests for the `rtk.ignite` module.
"""

import pytest

# torch
import torch
from torch.utils.data import DataLoader

# ignite
from ignite.engine import Engine, create_supervised_trainer, create_supervised_evaluator

# ignite.contrib
from ignite.contrib.handlers import ProgressBar

# rtk
from rtk import datasets, models
from rtk.config import (
    Configuration,
    DatasetConfiguration,
    DiffusionModelConfiguration,
    IgniteConfiguration,
    JobConfiguration,
    ModelConfiguration,
)
from rtk.ignite import (
    create_diffusion_model_engines,
    create_default_trainer_args,
    create_metrics,
    prepare_run,
    prepare_diffusion_run,
)
from rtk.utils import hydra_instantiate

MAX_EPOCHS = 3
EPOCH_LENGTH = 32
TRAINER_RUN_KWARGS = {"epoch_length": EPOCH_LENGTH, "max_epochs": MAX_EPOCHS}


class TestIgnite:
    @pytest.fixture
    def loaders(self, test_cfg: Configuration):
        """Fixture for the train loader."""
        dataset_cfg: DatasetConfiguration = test_cfg.datasets
        job_cfg: JobConfiguration = test_cfg.job
        use_transforms = job_cfg.use_transforms
        transform = datasets.create_transforms(test_cfg, use_transforms=use_transforms)
        val_transform = datasets.create_transforms(test_cfg, use_transforms=False)
        _datasets = datasets.instantiate_image_dataset(
            cfg=test_cfg, transform=transform
        )
        _train_dataset = _datasets[0]
        train_val_test_split_dict = datasets.instantiate_train_val_test_datasets(
            cfg=test_cfg,
            dataset=_train_dataset,
            train_transforms=transform,
            eval_transforms=val_transform,
        )
        train_dataset = train_val_test_split_dict["train"]
        train_loader: DataLoader = hydra_instantiate(
            cfg=dataset_cfg.dataloader,
            dataset=train_dataset,
            pin_memory=torch.cuda.is_available(),
            shuffle=True,
        )
        val_dataset = train_val_test_split_dict["val"]
        val_loader = hydra_instantiate(
            cfg=dataset_cfg.dataloader,
            dataset=val_dataset,
            pin_memory=torch.cuda.is_available(),
            shuffle=True,
        )
        test_dataset = _datasets[1]
        test_loader = hydra_instantiate(
            cfg=dataset_cfg.dataloader,
            dataset=test_dataset,
            pin_memory=torch.cuda.is_available(),
            shuffle=True,
        )
        return train_loader, val_loader, test_loader

    def test_create_default_trainer_args(self, test_cfg: Configuration):
        """Test the `rtk.ignite.create_default_trainer_args` function."""
        trainer_args = create_default_trainer_args(test_cfg)
        trainer = create_supervised_trainer(**trainer_args)
        assert trainer is not None

    def test_create_metrics(self, test_cfg: Configuration):
        """Test the `rtk.ignite.create_metrics` function."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = models.instantiate_criterion(test_cfg, device=device)
        metrics = create_metrics(cfg=test_cfg, criterion=criterion)
        assert metrics is not None

    @pytest.mark.requires_live_test
    def test_prepare_run(self, test_cfg: Configuration, loaders: tuple):
        """Test the `rtk.ignite.prepare_run` function."""
        device = torch.device(test_cfg.job.device)
        trainer, _ = prepare_run(cfg=test_cfg, loaders=loaders, device=device)

        # run trainer
        state = trainer.run(data=loaders[0], **TRAINER_RUN_KWARGS)
        assert len(state.batch) > 0

    # @pytest.mark.diffusion
    # def test_prepare_diffusion_run(self, test_cfg: Configuration, loaders: tuple):
    #     """Test the `rtk.ignite.prepare_diffusion_run` function."""

    #     dataset_cfg: DatasetConfiguration = test_cfg.datasets
    #     device = torch.device(test_cfg.job.device)
    #     train_loader = loaders[0]
    #     val_loader = loaders[1]
    #     new_loaders = [train_loader, val_loader]
    #     trainer, _ = prepare_diffusion_run(
    #         cfg=test_cfg, loaders=new_loaders, device=device
    #     )

    #     # run trainer
    #     trainer.run()
    #     assert True
