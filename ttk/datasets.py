# imports
import os
import pandas as pd
from copy import deepcopy
from hydra.utils import instantiate

# sklearn
from sklearn.model_selection import train_test_split

# torch
import torch
from torch.utils.data import Subset

# monai
import monai
import monai.transforms as monai_transforms

# teddytoolkit
from ttk.config import Configuration, JobConfiguration, DatasetConfiguration
from ttk.utils import get_logger, hydra_instantiate

logger = get_logger(__name__)


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
    logger.info("Creating transforms...")
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
            transform_fn = instantiate(transform)
            _ret_transforms.append(transform_fn)

        # always convert to tensor at the end
        _ret_transforms.append(monai_transforms.ToTensor())
        return _ret_transforms

    ret_transforms = __get_monai_transforms(transforms)

    return monai_transforms.Compose(ret_transforms)


def _filter_scan_paths(
    filter_function: callable, scan_paths: list, exclude: list = ["_mask.nii.gz"]
):
    """
    Filter a list of scan paths using a filter function.

    ## Args
    * `filter_function` (`callable`): The filter function to use.
    * `scan_paths` (`list`): The list of scan paths to filter.
    * `exclude` (`list`, optional): The list of strings to exclude from the scan paths. Defaults to `["_mask.nii.gz"]`.
    """
    filtered_scan_paths = [
        scan_path
        for scan_path in scan_paths
        if filter_function(scan_path) and not any(x in scan_path for x in exclude)
    ]
    return filtered_scan_paths

# TODO: Match IDs BEFORE pairing scan paths and labels to fix `ValueError: Found input variables with inconsistent numbers of samples: [578, 619]`
def instantiate_image_dataset(cfg: Configuration, **kwargs):
    """
    Instantiates a MONAI image dataset given a hydra configuration. This uses the `hydra.utils.instantiate` function to instantiate the dataset from the MONAI python package.

    ## Args
    * `dataset_cfg` (`DatasetConfiguration`): The dataset configuration.
    ## Returns
    * `monai.data.Dataset`: The instantiated dataset.
    """
    dataset_cfg: DatasetConfiguration = cfg.datasets
    # index = cfg.index
    target = cfg.target
    scan_data = dataset_cfg.scan_data
    patient_data = dataset_cfg.patient_data

    # get the scan paths
    scan_paths = [os.path.join(scan_data, f) for f in os.listdir(scan_data)]
    # get the patient dataframe
    patient_df = pd.read_excel(patient_data)
    labels = patient_df[target].values
    filtered_scan_paths = _filter_scan_paths(
        filter_function=lambda x: x.split("/")[-1], scan_paths=scan_paths
    )
    dataset: monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=filtered_scan_paths,
        labels=labels,
        **kwargs,
    )
    return dataset


def instantiate_train_val_test_datasets(
    cfg: Configuration, dataset: monai.data.ImageDataset, **kwargs
):
    """
    Create train/test splits for the data.
    """
    logger.info("Creating train/val/test splits...")
    job_cfg: JobConfiguration = cfg.job
    dataset_cfg: DatasetConfiguration = cfg.datasets
    train_val_test_split_dict = {}
    use_transforms = job_cfg.use_transforms
    train_transforms = create_transforms(
        dataset_cfg=dataset_cfg, use_transforms=use_transforms
    )
    eval_transforms = create_transforms(dataset_cfg=dataset_cfg, use_transforms=False)

    # create the train/test splits
    X = dataset.image_files
    y = dataset.labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, **job_cfg.train_test_split
    )
    ## create test dataset
    test_dataset: monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=X_test,
        labels=y_test,
        transform=eval_transforms,
        **kwargs,
    )
    train_val_test_split_dict["test_dataset"] = test_dataset
    ## create the train/val splits
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, stratify=y_train, **job_cfg.train_val_split
    )
    ## create train and val datasets
    train_dataset: monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=X_train,
        labels=y_train,
        transform=train_transforms,
        **kwargs,
    )
    train_val_test_split_dict["train_dataset"] = train_dataset
    val_dataset: monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=X_val,
        labels=y_val,
        transform=eval_transforms,
        **kwargs,
    )
    train_val_test_split_dict["val_dataset"] = val_dataset
    logger.info("Train/val/test splits created.")
    return train_val_test_split_dict
