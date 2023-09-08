"""
Tests for the `ttk.ignite` module.
"""

import pytest

# torch
import torch
from torch.utils.data import DataLoader

# ignite
from ignite.engine import Engine, create_supervised_trainer, create_supervised_evaluator

# ignite.contrib
from ignite.contrib.handlers import ProgressBar

# ttk
from ttk import datasets, models
from ttk.config import (
    Configuration,
    DatasetConfiguration,
    DiffusionModelConfiguration,
    IgniteConfiguration,
    JobConfiguration,
    ModelConfiguration,
)
from ttk.ignite import (
    create_diffusion_model_engines,
    create_default_trainer_args,
    create_metrics,
    prepare_run,
)
from ttk.utils import hydra_instantiate

MAX_EPOCHS = 3
EPOCH_LENGTH = None
TRAINER_RUN_KWARGS = {"epoch_length": EPOCH_LENGTH, "max_epochs": MAX_EPOCHS}


class TestIgnite:
    @pytest.fixture
    def loaders(self, test_cfg: Configuration):
        """Fixture for the train loader."""
        dataset_cfg: DatasetConfiguration = test_cfg.datasets
        job_cfg: JobConfiguration = test_cfg.job
        transform = datasets.create_transforms(
            dataset_cfg, use_transforms=job_cfg.use_transforms
        )
        dataset = datasets.instantiate_image_dataset(cfg=test_cfg, transform=transform)
        _train_dataset = dataset[0]
        train_val_test_split_dict = datasets.instantiate_train_val_test_datasets(
            cfg=test_cfg, dataset=_train_dataset
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
        return train_loader, val_loader

    def test_create_default_trainer_args(self, test_cfg: Configuration, loaders: tuple):
        """Test the `ttk.ignite.create_default_trainer_args` function."""
        trainer_args = create_default_trainer_args(test_cfg)
        trainer = create_supervised_trainer(**trainer_args)
        ProgressBar().attach(trainer)

        # run trainer
        train_loader = loaders[0]
        state = trainer.run(data=train_loader, **TRAINER_RUN_KWARGS)
        assert state is not None

    def test_create_metrics(self, test_cfg: Configuration):
        """Test the `ttk.ignite.create_metrics` function."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = models.instantiate_criterion(test_cfg.models, device=device)
        metrics = create_metrics(cfg=test_cfg, criterion=criterion)
        assert metrics is not None

    def test_prepare_run(self, test_cfg: Configuration, loaders: tuple):
        """Test the `ttk.ignite.prepare_run` function."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer, _ = prepare_run(cfg=test_cfg, loaders=loaders, device=device)

        # run trainer
        state = trainer.run(data=loaders[0], **TRAINER_RUN_KWARGS)
        assert state is not None

    @pytest.mark.diffusion
    def test_create_diffusion_model_engines(
        self, test_cfg: Configuration, train_loader: DataLoader
    ):
        """Test the `ttk.ignite.create_diffusion_model_engines` function."""

        ## create engines
        # _callback_dict = {}
        engine_dict = create_diffusion_model_engines(
            cfg=test_cfg, train_loader=train_loader
        )
        trainer = engine_dict["trainer"]

        # run trainer
        state = trainer.run(data=train_loader, **TRAINER_RUN_KWARGS)
        assert state is not None
