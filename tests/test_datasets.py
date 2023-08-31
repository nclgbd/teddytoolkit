"""Tests for the `ttk.datasets` module."""
import math
import os
import numpy as np
import pandas as pd
from omegaconf import DictConfig

# ttk
from ttk import DEFAULT_DATA_PATH, datasets
from ttk.config import Configuration, DatasetConfiguration, JobConfiguration


class TestDatasets:
    def test_create_transforms(self, test_cfg: Configuration):
        """Tests the `ttk.datasets.create_transforms` function."""
        transforms = datasets.create_transforms(dataset_cfg=test_cfg.datasets)
        # assert transforms were created
        assert transforms is not None

    def test_instantiate_image_dataset(self, test_cfg: Configuration):
        """Tests the `ttk.datasets.instantiate_image_dataset` function."""
        dataset_cfg: DatasetConfiguration = test_cfg.datasets
        transform_cfg: DictConfig = dataset_cfg.transforms
        transform = datasets.create_transforms(dataset_cfg)
        dataset = datasets.instantiate_image_dataset(
            cfg=test_cfg, save_metadata=True, transform=transform
        )
        train_dataset, test_dataset = dataset[0], dataset[1]
        # assert ixi dataset was created
        assert train_dataset is not None
        assert len(train_dataset) == 5232
        assert len(test_dataset) == 624
        assert any(train_dataset.labels)
        # attempt retrieval of sample
        scan, label = train_dataset[0]
        assert scan is not None
        assert type(label) == np.int64 or type(label) == int
        # check shape of sample
        scan_shape = scan.shape[1:]
        assert scan_shape == tuple(transform_cfg["load"][-1]["spatial_size"])

        ## assert ixi dataset was created
        # assert dataset is not None
        # assert any(dataset.labels)
        # assert len(dataset) == 538
        # # attempt retrieval of sample
        # scan, label = dataset[0]
        # assert scan is not None
        # assert type(label) == np.int64
        # # check shape of sample
        # scan_shape = scan.shape[1:]
        # assert scan_shape == tuple(transform_cfg["load"][-1]["spatial_size"])

    def test_instantiate_train_val_test_datasets(self, test_cfg: Configuration):
        """Tests the `ttk.datasets.instantiate_train_val_test_datasets` function."""
        dataset_cfg: DatasetConfiguration = test_cfg.datasets
        job_cfg: JobConfiguration = test_cfg.job

        transform = datasets.create_transforms(dataset_cfg)
        dataset = datasets.instantiate_image_dataset(cfg=test_cfg, transform=transform)
        train_val_test_split_dict = datasets.instantiate_train_val_test_datasets(
            cfg=test_cfg, save_metadata=True, dataset=dataset
        )

        assert train_val_test_split_dict is not None
        if job_cfg.perform_validation:
            assert len(set(train_val_test_split_dict.keys())) == 3
        else:
            assert len(set(train_val_test_split_dict.keys())) == 2

        test_dataset = train_val_test_split_dict["test"]
        assert test_dataset is not None
        test_dataset_len = len(test_dataset)
        test_proportion = job_cfg.train_test_split["test_size"] * len(dataset)
        assert (test_dataset_len == math.ceil(test_proportion)) or (
            test_dataset_len == math.floor(test_proportion)
        )
