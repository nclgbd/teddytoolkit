"""Tests for the `ttk.datasets` module."""
import numpy as np
from omegaconf import DictConfig

from ttk import datasets
from ttk.config import Configuration, DatasetConfiguration


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
        dataset = datasets.instantiate_image_dataset(cfg=test_cfg, transform=transform)
        # assert dataset was created
        assert dataset is not None
        assert any(dataset.labels)
        assert len(dataset) == 578
        # attempt retrieval of sample
        scan, label = dataset[0]
        assert scan is not None
        assert type(label) == np.int64
        # check shape of sample
        scan_shape = scan.shape[1:]
        assert scan_shape == tuple(transform_cfg["load"][-1]["spatial_size"])

    def test_instantiate_train_val_test_datasets(self, test_cfg: Configuration):
        """Tests the `ttk.datasets.instantiate_train_val_test_datasets` function."""
        dataset_cfg: DatasetConfiguration = test_cfg.datasets
        transform_cfg: DictConfig = dataset_cfg.transforms
        # transform = datasets.create_transforms(dataset_cfg)
        dataset = datasets.instantiate_image_dataset(cfg=test_cfg)
        train_val_test_split_dict = datasets.instantiate_train_val_test_datasets(
            cfg=test_cfg, dataset=dataset
        )
        assert train_val_test_split_dict is not None
