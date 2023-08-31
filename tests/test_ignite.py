"""
Tests for the `ttk.ignite` module.
"""

import pytest

# torch
import torch
from torch.utils.data import DataLoader

# ignite
from ignite.engine import Engine, create_supervised_trainer

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
from ttk.ignite import create_diffusion_model_engines, create_default_trainer_args
from ttk.utils import hydra_instantiate

MAX_EPOCHS = 1
EPOCH_LENGTH = 10
TRAINER_RUN_KWARGS = {"epoch_length": EPOCH_LENGTH, "max_epochs": MAX_EPOCHS}


class TestIgnite:
    @pytest.fixture
    def train_loader(self, test_cfg: Configuration):
        """Fixture for the train loader."""
        dataset_cfg: DatasetConfiguration = test_cfg.datasets
        transform = datasets.create_transforms(dataset_cfg)
        dataset = datasets.instantiate_image_dataset(cfg=test_cfg, transform=transform)
        train_val_test_split_dict = datasets.instantiate_train_val_test_datasets(
            cfg=test_cfg, dataset=dataset
        )
        train_dataset = train_val_test_split_dict["train"]
        train_loader: DataLoader = hydra_instantiate(
            cfg=dataset_cfg.dataloader,
            dataset=train_dataset,
            pin_memory=torch.cuda.is_available(),
        )
        # val_dataset = train_val_test_split_dict["val"]
        # val_loader = hydra_instantiate(
        #     cfg=dataset_cfg.dataloader,
        #     dataset=val_dataset,
        #     pin_memory=torch.cuda.is_available(),
        # )
        return train_loader

    def test_create_default_trainer_args(
        self, test_cfg: Configuration, train_loader: DataLoader
    ):
        """Test the `ttk.ignite.create_default_trainer_args` function."""

        trainer_args = create_default_trainer_args(test_cfg)
        trainer = create_supervised_trainer(deterministic=True, **trainer_args)
        ProgressBar().attach(trainer)

        # run trainer
        state = trainer.run(data=train_loader, **TRAINER_RUN_KWARGS)
        assert state is not None

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
