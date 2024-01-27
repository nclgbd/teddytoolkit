import hydra
from copy import deepcopy

# torch
import torch

# monai
import monai.transforms as monai_transforms

# rtk
from rtk import *
from rtk.config import *


def create_transforms(
    cfg: Configuration = None,
    dataset_cfg: DatasetConfiguration = None,
    use_transforms: bool = None,
    transform_dicts: dict = None,
    **kwargs,
):
    """
    Get transforms for the model based on the model configuration.
    ## Args
    * `cfg` (`Configuration`, optional): The configuration. Defaults to `None`.
    * `dataset_cfg` (`DatasetConfiguration`, optional): The dataset configuration. Defaults to `None`.
    * `use_transforms` (`bool`, optional): Whether or not to use the transforms. Defaults to `False`.
    * `transform_dicts` (`dict`, optional): The dictionary of transforms to use. Defaults to `None`.
    * `mode` (`str`, optional): The mode to use. Add in kwargs.
    ## Returns
    * `torchvision.transforms.Compose`: The transforms for the model in the form of a `torchvision.transforms.Compose`
    object.
    """
    dataset_cfg = cfg.datasets if cfg is not None else dataset_cfg
    use_transforms = (
        use_transforms
        if use_transforms is not None
        else cfg.job.get("use_transforms", False)
    )
    if use_transforms:
        logger.info("Creating 'train' transforms...")
    else:
        logger.info("Creating 'eval' transforms...")

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
            logger.debug(
                "Adding transform: '{}'".format(transform["_target_"].split(".")[-1])
            )
            transform_fn = hydra.utils.instantiate(transform)
            _ret_transforms.append(transform_fn)

        # diffusion transforms
        if kwargs.get("mode", "") == "diffusion":
            if use_transforms:
                rand_lambda_transform = monai_transforms.RandLambdad(
                    keys=[LABEL_KEYNAME],
                    prob=0.15,
                    func=lambda x: -1 * torch.ones_like(x),
                )
                _ret_transforms.append(rand_lambda_transform)
            lambda_transform = monai_transforms.Lambdad(
                keys=[LABEL_KEYNAME],
                func=lambda x: torch.tensor(x, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0),
            )
            _ret_transforms.append(lambda_transform)
            # _ret_transforms.appent(monai_transforms.EnsureType(dtype=torch.float32))

        # always convert to tensor at the end
        _ret_transforms.append(monai_transforms.ToTensor())
        return _ret_transforms

    ret_transforms = __get_monai_transforms(transforms)

    return monai_transforms.Compose(ret_transforms)
