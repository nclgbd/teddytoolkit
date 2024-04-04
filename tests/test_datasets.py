"""Tests for the `rtk.datasets` module."""
import math
import os
import numpy as np
import pandas as pd
from omegaconf import DictConfig

# torch
import torch

# monai
from monai.data import DataLoader
from monai.utils import first

# rtk
from rtk import (
    DEFAULT_DATA_PATH,
    datasets,
)
from rtk.datasets import (
    _IMAGE_KEYNAME,
    _LABEL_KEYNAME,
)
from rtk.config import ImageClassificationConfiguration, ImageDatasetConfiguration, JobConfiguration


class TestDatasets:
    def test_create_transforms(self, test_cfg: ImageClassificationConfiguration):
        """Tests the `rtk.datasets.create_transforms` function."""
        dataset_cfg: ImageDatasetConfiguration = test_cfg.datasets
        transforms = datasets.create_transforms(test_cfg, use_transforms=False)

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

    def test_instantiate_image_dataset(self, test_cfg: ImageClassificationConfiguration):
        """Tests the `rtk.datasets.instantiate_image_dataset` function."""
        dataset_cfg: ImageDatasetConfiguration = test_cfg.datasets
        transform_cfg: DictConfig = dataset_cfg.transforms
        transform = datasets.create_transforms(cfg=test_cfg)
        dataset = datasets.instantiate_image_dataset(
            cfg=test_cfg, save_metadata=True, transform=transform
        )
        train_dataset, test_dataset = dataset[0], dataset[1]
        # assert dataset was created
        assert train_dataset is not None
        scan, label = first(train_dataset)
        assert isinstance(scan, torch.Tensor)
        assert type(label) == np.int64 or type(label) == int
        # check shape of sample
        scan_shape = scan.shape[1:]
        assert scan_shape == tuple(transform_cfg["load"][-1]["spatial_size"])

    def test_instantiate_train_val_test_datasets(self, test_cfg: ImageClassificationConfiguration):
        """Tests the `rtk.datasets.instantiate_train_val_test_datasets` function."""
        dataset_cfg: ImageDatasetConfiguration = test_cfg.datasets
        job_cfg: JobConfiguration = test_cfg.job

        transform = datasets.create_transforms(test_cfg)
        dataset = datasets.instantiate_image_dataset(cfg=test_cfg, transform=transform)
        train_dataset = dataset[0]
        use_val = job_cfg.perform_validation
        train_val_test_split_dict = datasets.instantiate_train_val_test_datasets(
            cfg=test_cfg,
            dataset=train_dataset,
            train_transforms=transform,
            eval_transforms=transform,
            save_metadata=True,
        )

        assert train_val_test_split_dict is not None
        if use_val and dataset_cfg.extension == ".nii.gz":
            assert len(set(train_val_test_split_dict.keys())) == 3
        elif use_val:
            assert len(set(train_val_test_split_dict.keys())) == 2

        train_dataset = train_val_test_split_dict["train"]
        assert train_dataset is not None

        # test `rtk.datasets.resample_to_value`
        if dataset_cfg.preprocessing.use_sampling:
            preprocessing_cfg = dataset_cfg.preprocessing
            assert len(train_dataset) == preprocessing_cfg.sampling_method[
                "sample_to_value"
            ] * len(dataset_cfg.labels)

    def test_transform_image_dataset_to_cache_dataset(self, test_cfg):
        """Tests the `rtk.datasets.transform_image_dataset_to_cache_dataset` function."""
        dataset_cfg: ImageDatasetConfiguration = test_cfg.datasets
        transform_cfg = dataset_cfg.transforms
        transform = datasets.create_transforms(test_cfg)
        _datasets = datasets.instantiate_image_dataset(
            cfg=test_cfg, transform=transform
        )
        train_dataset = _datasets[0]
        train_dataset = datasets.convert_image_dataset(train_dataset)
        assert train_dataset is not None

        check_data = first(train_dataset)
        scan, label = check_data[_IMAGE_KEYNAME], check_data[_LABEL_KEYNAME]
        assert isinstance(scan, torch.Tensor)
        assert type(label) == np.int64 or type(label) == int
        # check shape of sample
        scan_shape = scan.shape[1:]
        assert scan_shape == torch.Size([dataset_cfg.dim, dataset_cfg.dim])
