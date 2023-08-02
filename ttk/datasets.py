# imports
import os
import pandas as pd
from copy import deepcopy
from hydra.utils import instantiate

# torch
import torch

# monai
import monai
import monai.transforms as monai_transforms

# teddytoolkit
from ttk.config import Configuration, DatasetConfiguration


def create_transforms(
    dataset_cfg: DatasetConfiguration = None,
    use_transforms: bool = False,
    transform_dicts: dict = None,
    **kwargs,
):
    """
    Get transforms for the model based on the model configuration.
    ## Args
    * `model_config` (`TorchModelConfiguration`, optional): The model configuration. Defaults to `None`.
    * `use_transforms` (`bool`, optional): Whether or not to use the transforms. Defaults to `False`.
    * `transform_dicts` (`dict`, optional): The dictionary of transforms to use. Defaults to `None`.
    ## Returns
    * `torchvision.transforms.Compose`: The transforms for the model in the form of a `torchvision.transforms.Compose`
    object.
    """
    # console.log("Creating transforms...")
    transform_dicts: dict = (
        transform_dicts
        if dataset_cfg is None
        else dataset_cfg.get("transforms", transform_dicts)
    )

    if transform_dicts is None:
        return None

    # transforms specific to loading the data. These are always used
    transforms: list = deepcopy(transform_dicts["load"])

    # If we're using transforms, we need to load the training dictionaries as well
    if use_transforms:
        transforms += transform_dicts["train"]

    def __get_monai_transforms(
        transforms: list,
    ):
        _ret_transforms = []
        for transform in transforms:
            # console.log(
            #     "Adding transform: '{}'".format(transform["_target_"].split(".")[-1])
            # )
            transform_fn = instantiate(transform)
            _ret_transforms.append(transform_fn)

        # always convert to tensor at the end
        _ret_transforms.append(monai_transforms.ToTensor())
        return _ret_transforms

    ret_transforms = __get_monai_transforms(transforms)

    return monai_transforms.Compose(ret_transforms)


def instantiate_monai_dataset(dataset_cfg: DatasetConfiguration, **kwargs):
    """
    Instantiates a MONAI dataset given a hydra configuration. This uses the `hydra.utils.instantiate` function to instantiate the dataset from the MONAI python package.

    ## Args
    * `dataset_cfg` (`DatasetConfiguration`): The dataset configuration.
    ## Returns
    * `monai.data.Dataset`: The instantiated dataset.
    """
    scan_data = dataset_cfg.scan_data
    scan_paths = [os.path.join(scan_data, f) for f in os.listdir(scan_data)]
    dataset: monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=scan_paths,
        **kwargs,
    )
    return dataset
