import hydra
import numpy as np
import pandas as pd
import skimage
from copy import deepcopy
from PIL import Image

# sklean
from imblearn.under_sampling import RandomUnderSampler

# azureml
from azureml.core import Workspace

# monai
import monai.transforms as monai_transforms

# rtk
from rtk import *
from rtk.config import *
from rtk.utils import load_patient_dataset, login


def resample_to_value(
    cfg: BaseConfiguration,
    metadata: pd.DataFrame,
    dataset_labels: list = [],
    sampling_strategy: float = 1.0,
    **kwargs,
):
    """
    Resample a dataset by duplicating the data.
    `sampling_strategy` is the ratio of the number of samples in the minority class to the number of samples in the majority class after resampling.
    """
    dataset_cfg = cfg.datasets
    index = dataset_cfg.index
    preprocessing_cfg = dataset_cfg.preprocessing
    positive_class = preprocessing_cfg.positive_class

    # we have to recheck the size of the data since the data has already been split into training
    subsample_size = len(metadata[metadata[positive_class] == 1])
    sample_to_value: int = preprocessing_cfg.sampling_method.get(
        "sample_to_value", kwargs.get("sample_to_value", subsample_size)
    )
    metadata_copy = deepcopy(metadata).reset_index()
    new_metadata = pd.DataFrame(columns=metadata_copy.columns)
    if sampling_strategy == 1.0:
        for label in dataset_labels:
            # have the data resample from Pneumonia
            # query = metadata_copy[[label] + minority_class_names]
            class_subset: pd.DataFrame = metadata_copy[metadata_copy[label] == 1]

            if sample_to_value - class_subset.shape[0] <= 0:
                class_subset = class_subset.sample(
                    n=sample_to_value, replace=False, random_state=cfg.random_state
                )

            else:
                class_subset = class_subset.sample(
                    n=sample_to_value, replace=True, random_state=cfg.random_state
                )

            new_metadata = pd.concat(
                [
                    new_metadata,
                    class_subset,
                ],
            )
            # assert len(new_metadata) <= sample_to_value * len(dataset_labels)
    else:
        # perform undersampling using v1.0 labels
        # sampler = RandomUnderSampler(random_state=cfg.random_state)
        positive_indices = metadata_copy[metadata_copy[positive_class] == 1].index
        positive_samples = metadata_copy.loc[positive_indices]
        negative_samples = metadata_copy.drop(positive_indices)
        negative_samples = negative_samples.sample(
            n=len(positive_samples), replace=False, random_state=cfg.random_state
        )
        new_metadata = pd.concat([positive_samples, negative_samples])

    class_counts = dict()
    for label in dataset_labels:
        class_counts[label] = len(new_metadata[new_metadata[label] == 1])

    logger.info(f"New class counts (with overlap):\n{class_counts}")

    return new_metadata.reset_index(drop=True)


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


def load_metadata(
    index: str, return_workspace=False, ws: Workspace = None, *args, **kwargs
):
    if ws is None:
        ws: Workspace = login()
    metadata: pd.DataFrame = load_patient_dataset(ws, *args, **kwargs).set_index(index)

    if return_workspace:
        return metadata, ws
    return metadata
