"""Tests for the `rtk.datasets` module."""
import math
import os
import numpy as np
import pandas as pd
from omegaconf import DictConfig

# monai
from monai.data import DataLoader
from monai.utils import first

# rtk
from rtk import (
    DEFAULT_DATA_PATH,
    datasets,
)
from rtk.datasets import (
    _CHEST_XRAY_TRAIN_DATASET_SIZE,
    _CHEST_XRAY_TEST_DATASET_SIZE,
    _IXI_MRI_DATASET_SIZE,
)
from rtk.config import Configuration, DatasetConfiguration, JobConfiguration


class TestDatasets:
    def test_create_transforms(self, test_cfg: Configuration):
        """Tests the `rtk.datasets.create_transforms` function."""
        dataset_cfg: DatasetConfiguration = test_cfg.datasets
        transforms = datasets.create_transforms(test_cfg)

        # without transforms
        assert transforms is not None
        load_transforms = dataset_cfg.transforms["load"]
        assert len(transforms.transforms) == len(load_transforms) + 1

        # with transforms
        transforms = datasets.create_transforms(test_cfg, use_transforms=True)
        assert transforms is not None
        eval_transforms = dataset_cfg.transforms["train"]
        assert (
            len(transforms.transforms)
            == len(load_transforms) + len(eval_transforms) + 1
        )

        # test diffusion transforms
        # TODO: properly test this
        transforms = datasets.create_transforms(
            test_cfg,
            mode="diffusion",
            use_transforms=True,
        )
        # assert transforms were created
        assert transforms is not None

    def test_instantiate_image_dataset(self, test_cfg: Configuration):
        """Tests the `rtk.datasets.instantiate_image_dataset` function."""
        dataset_cfg: DatasetConfiguration = test_cfg.datasets
        transform_cfg: DictConfig = dataset_cfg.transforms
        transform = datasets.create_transforms(cfg=test_cfg)
        dataset = datasets.instantiate_image_dataset(
            cfg=test_cfg, save_metadata=True, transform=transform
        )
        train_dataset, test_dataset = dataset[0], dataset[1]
        # assert dataset was created
        assert train_dataset is not None

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
        # train_dataset = datasets.transform_image_dataset_to_cache_dataset(train_dataset)
        scan, label = next(iter(DataLoader(train_dataset)))
        assert scan is not None
        assert type(label) == np.int64 or type(label) == int
        # check shape of sample
        scan_shape = scan.shape[1:]
        assert scan_shape == tuple(transform_cfg["load"][-1]["spatial_size"])

    def test_instantiate_train_val_test_datasets(self, test_cfg: Configuration):
        """Tests the `rtk.datasets.instantiate_train_val_test_datasets` function."""
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

        # test `rtk.datasets.resample_to_value`
        if dataset_cfg.use_sampling:
            num_classes = len(dataset_cfg.labels)
            assert len(train_dataset) == dataset_cfg.sample_to_value * num_classes

    def test_transform_image_dataset_to_cache_dataset(self, test_cfg):
        """Tests the `rtk.datasets.transform_image_dataset_to_cache_dataset` function."""
        dataset_cfg: DatasetConfiguration = test_cfg.datasets
        transform = datasets.create_transforms(dataset_cfg)
        dataset = datasets.instantiate_image_dataset(cfg=test_cfg, transform=transform)
        train_dataset = dataset[0]
        train_dataset_dict = datasets.convert_image_dataset(train_dataset)
        assert train_dataset_dict is not None

    def test_preprocess_dataset(self, test_cfg: Configuration):
        """Tests the `rtk.datasets.preprocess_dataset` function."""
        dataset_cfg: DatasetConfiguration = test_cfg.datasets
        transform = datasets.create_transforms(dataset_cfg)
        dataset = datasets.instantiate_image_dataset(cfg=test_cfg, transform=transform)
        train_dataset = dataset[0]
        new_train_dataset = datasets.convert_image_dataset(train_dataset)
        new_dataset = datasets.preprocess_dataset(
            dataset=new_train_dataset, cfg=test_cfg
        )
        assert new_dataset is not None
