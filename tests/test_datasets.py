from omegaconf import DictConfig

from ttk import datasets
from ttk.config import Configuration, DatasetConfiguration


class TestDatasets:
    def test_create_transforms(self, test_cfg: Configuration):
        """Tests the `ttk.create_transforms` function."""
        transforms = datasets.create_transforms(dataset_cfg=test_cfg.datasets)
        # assert transforms were created
        assert transforms is not None

    def test_instantiate_image_dataset(self, test_cfg: Configuration):
        """Tests the `ttk.instantiate_image_dataset` function."""
        dataset_cfg: DatasetConfiguration = test_cfg.datasets
        transform_cfg: DictConfig = dataset_cfg.transforms
        transform = datasets.create_transforms(dataset_cfg)
        dataset = datasets.instantiate_image_dataset(
            dataset_cfg=dataset_cfg, transform=transform
        )
        # assert dataset was created
        assert dataset is not None
        assert len(dataset) == 578
        # attempt retrieval of sample
        sample = dataset[0]
        assert sample is not None
        # check shape of sample
        sample_shape = sample.shape[1:]
        assert sample_shape == tuple(transform_cfg["load"][-1]["spatial_size"])
