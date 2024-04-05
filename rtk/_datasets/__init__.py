import hydra
import numpy as np
import pandas as pd
import skimage
from copy import deepcopy
from PIL import Image

# monai
import monai.transforms as monai_transforms

# rtk
from rtk import *
from rtk.config import *
from rtk.utils import load_patient_dataset, login


def create_transforms(
    cfg: ImageConfiguration = None,
    dataset_cfg: ImageDatasetConfiguration = None,
    use_transforms: bool = None,
    transform_dicts: dict = None,
):
    dataset_cfg = cfg.datasets if cfg is not None else dataset_cfg
    use_transforms = (
        use_transforms
        if use_transforms is not None
        else cfg.get("use_transforms", False)
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

    def __get_transforms(
        transforms: list,
    ):
        _ret_transforms = []
        for transform in transforms:
            logger.debug(
                "Adding transform: '{}'".format(transform["_target_"].split(".")[-1])
            )
            transform_fn = hydra.utils.instantiate(transform)
            _ret_transforms.append(transform_fn)

        # always convert to tensor at the end
        _ret_transforms.append(monai_transforms.ToTensor())
        return _ret_transforms

    ret_transforms = __get_transforms(transforms)

    return monai_transforms.Compose(ret_transforms)


def load_metadata(index: str, *args, **kwargs) -> pd.DataFrame:
    ws = login()
    metadata: pd.DataFrame = load_patient_dataset(ws, *args, **kwargs).set_index(index)
    return metadata
