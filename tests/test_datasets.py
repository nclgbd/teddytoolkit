from omegaconf import OmegaConf

from ttk import datasets
from ttk.config import Configuration, DatasetConfiguration


class TestDatasets:
    def test_instantiate_monai_dataset(self, test_cfg: Configuration):
        datasets.instantiate_monai_dataset(test_cfg.dataset)
        transforms = datasets.create_transforms(test_cfg.dataset)
        dataset = datasets.instantiate_monai_dataset(test_cfg.dataset, transforms=transforms)
        assert dataset is not None
        assert len(dataset) == 578

    def test_create_transforms(self, test_cfg: Configuration):
        transforms = datasets.create_transforms(test_cfg.dataset)
        assert transforms is not None
