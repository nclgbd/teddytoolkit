"""
Tests for the `ttk.ignite` module.
"""

import pytest

# torch
import torch

# ignite
from ignite.engine import Engine

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
)
from ttk.ignite import create_diffusion_model_engines
from ttk.utils import hydra_instantiate

MAX_EPOCHS = 1


class TestIgnite:
    def test_create_diffusion_model_engines(self, test_cfg: Configuration):
        """Test the `ttk.ignite.create_diffusion_model_engines` function."""
        # create data
        dataset_cfg: DatasetConfiguration = test_cfg.datasets
        job_cfg: JobConfiguration = test_cfg.job

        transform = datasets.create_transforms(dataset_cfg)
        dataset = datasets.instantiate_image_dataset(cfg=test_cfg, transform=transform)
        train_loader = hydra_instantiate(cfg=dataset_cfg.dataloader, dataset=dataset)
        # train_val_test_split_dict = datasets.instantiate_train_val_test_datasets(
        #     cfg=test_cfg, dataset=dataset
        # )
        # train_dataset = train_val_test_split_dict["train"]
        # train_loader = hydra_instantiate(
        #     cfg=dataset_cfg.dataloader, dataset=train_dataset
        # )
        # val_dataset = train_val_test_split_dict["val"]
        # val_loader = hydra_instantiate(cfg=dataset_cfg.dataloader, dataset=val_dataset)

        # create engines
        _callback_dict = {}
        engine_dict = create_diffusion_model_engines(
            cfg=test_cfg, train_loader=train_loader
        )
        trainer = engine_dict["trainer"]

        # run trainer
        state = trainer.run(data=train_loader, max_epochs=MAX_EPOCHS)
        assert state is not None
