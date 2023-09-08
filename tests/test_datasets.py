"""Tests for the `ttk.datasets` module."""
import math
import os
import numpy as np
import pandas as pd
from omegaconf import DictConfig

# ttk
from ttk import (
    DEFAULT_DATA_PATH,
    datasets,
)
from ttk.datasets import (
    _CHEST_XRAY_TRAIN_DATASET_SIZE,
    _CHEST_XRAY_TEST_DATASET_SIZE,
    _IXI_MRI_DATASET_SIZE,
)
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
        # assert dataset was created
        assert train_dataset is not None
        assert any(train_dataset.labels)

        # check length of dataset
        if test_cfg.job.perform_validation and dataset_cfg.extension == ".nii.gz":
            assert len(dataset) == _IXI_MRI_DATASET_SIZE
        elif test_cfg.job.perform_validation and dataset_cfg.extension == ".jpeg":
            # see if resampling was applied properly
            resample_value: int = dataset_cfg.get("resample_value", 1)
            if resample_value > 1:
                labels: np.array = train_dataset.labels
                assert len(np.unique(labels)) == len(dataset_cfg.labels)
            else:
                assert len(train_dataset) == _CHEST_XRAY_TRAIN_DATASET_SIZE

            assert len(test_dataset) == _CHEST_XRAY_TEST_DATASET_SIZE

        # attempt retrieval of sample
        scan, label = train_dataset[0]
        assert scan is not None
        assert type(label) == np.int64 or type(label) == int
        # check shape of sample
        scan_shape = scan.shape[1:]
        assert scan_shape == tuple(transform_cfg["load"][-1]["spatial_size"])

    def test_instantiate_train_val_test_datasets(self, test_cfg: Configuration):
        """Tests the `ttk.datasets.instantiate_train_val_test_datasets` function."""
        dataset_cfg: DatasetConfiguration = test_cfg.datasets
        job_cfg: JobConfiguration = test_cfg.job

        transform = datasets.create_transforms(dataset_cfg)
        dataset = datasets.instantiate_image_dataset(cfg=test_cfg, transform=transform)
        train_dataset = dataset[0]
        train_val_test_split_dict = datasets.instantiate_train_val_test_datasets(
            cfg=test_cfg, save_metadata=True, dataset=train_dataset
        )

        assert train_val_test_split_dict is not None
        if job_cfg.perform_validation and dataset_cfg.extension == ".nii.gz":
            assert len(set(train_val_test_split_dict.keys())) == 3
        elif job_cfg.perform_validation and dataset_cfg.extension == ".jpeg":
            assert len(set(train_val_test_split_dict.keys())) == 2

        train_dataset = train_val_test_split_dict["train"]
        assert train_dataset is not None
