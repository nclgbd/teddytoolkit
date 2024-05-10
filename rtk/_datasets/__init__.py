import hydra
import numpy as np
import pandas as pd
from copy import deepcopy

# azureml
from azureml.core import Workspace

# monai
import monai.transforms as monai_transforms

# rtk
from rtk import *
from rtk.config import *
from rtk.utils import load_patient_dataset, login


def get_class_counts(metadata: pd.DataFrame, dataset_labels: list):

    class_counts = dict()
    for label in dataset_labels:
        class_counts[label] = len(metadata[metadata[label] == 1])

    class_counts = pd.DataFrame.from_dict(
        class_counts, orient="index", columns=["Occurrences"]
    )
    return class_counts


def resample_to_value(
    metadata: pd.DataFrame,
    dataset_labels: list = [],
    sampling_strategy: str = "text_prompts",
    cfg: BaseConfiguration = None,
    **kwargs,
):
    """
    Resample a dataset by duplicating the data.
    `sampling_strategy` is the ratio of the number of samples in the minority class to the number of samples in the majority class after resampling.
    """
    dataset_cfg: DatasetConfiguration = kwargs.get("dataset_cfg", None)
    if dataset_cfg is None:
        dataset_cfg = cfg.datasets
    # index = dataset_cfg.index
    preprocessing_cfg = kwargs.get("preprocessing_cfg", None)
    if preprocessing_cfg is None:
        preprocessing_cfg = dataset_cfg.preprocessing
    positive_class = preprocessing_cfg.positive_class
    random_state = kwargs.get("random_state", 42)

    # we have to recheck the size of the data since the data has already been split into training
    subsample_size = len(metadata[metadata[positive_class] == 1])
    sample_to_value: int = preprocessing_cfg.sampling_method.get(
        "sample_to_value", kwargs.get("sample_to_value", subsample_size)
    )
    metadata_copy = deepcopy(metadata).reset_index()
    if sampling_strategy == "text_prompts":
        new_metadata = pd.DataFrame(columns=metadata_copy.columns)
        for label in dataset_labels:
            # have the data resample from Pneumonia
            # query = metadata_copy[[label] + minority_class_names]
            class_subset: pd.DataFrame = metadata_copy[metadata_copy[label] == 1]

            if sample_to_value - class_subset.shape[0] <= 0:
                class_subset = class_subset.sample(
                    n=sample_to_value, replace=False, random_state=random_state
                )

            else:
                class_subset = class_subset.sample(
                    n=sample_to_value, replace=True, random_state=random_state
                )

            new_metadata = pd.concat(
                [
                    new_metadata,
                    class_subset,
                ],
            )
    else:
        # create splits
        positive_indices = metadata_copy[metadata_copy[positive_class] == 1].index
        positive_samples = metadata_copy.loc[positive_indices]
        negative_samples = metadata_copy.drop(positive_indices)

        # resample positive class
        p_replace = True if len(positive_samples) < sample_to_value else False
        positive_samples = positive_samples.sample(
            n=sample_to_value, replace=p_replace, random_state=random_state
        )
        # resample negative class
        n_replace = True if len(negative_samples) < sample_to_value else False
        negative_samples = negative_samples.sample(
            n=sample_to_value, replace=n_replace, random_state=random_state
        )
        new_metadata = pd.concat([positive_samples, negative_samples]).reset_index(
            drop=True
        )

    class_counts = get_class_counts(new_metadata, dataset_labels)

    logger.debug(f"New class counts (with overlap):\n{class_counts}")

    return new_metadata


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
