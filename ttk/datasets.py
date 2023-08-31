# imports
import numpy as np
import os
import pandas as pd
import sys
from copy import deepcopy
from hydra.utils import instantiate
from rich import inspect
from tqdm import tqdm

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# torch
import torch
from torch.utils.data import Subset

# monai
import monai
import monai.transforms as monai_transforms
from monai.data import ImageDataset, ThreadDataLoader

# teddytoolkit
from ttk import DEFAULT_DATA_PATH
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


# TODO: pd.Series are saved in the dataframe for some reason, so we need to convert them to ints
def build_ixi_metadata_dataframe(
    cfg: Configuration, image_files: list, labels: list = None
):
    index_name = cfg.index
    target_name = cfg.target
    dataset_cfg: DatasetConfiguration = cfg.datasets
    patient_data = dataset_cfg.patient_data

    # create temporary dataset and dataloader to get the metadata
    patient_df = pd.read_excel(patient_data).set_index(index_name)
    tmp_dataset = ImageDataset(image_files=image_files, labels=labels, transform=None)
    tmp_dataloader = iter(
        ThreadDataLoader(
            tmp_dataset,
            batch_size=dataset_cfg.dataloader.batch_size,
            num_workers=dataset_cfg.dataloader.num_workers,
        )
    )

    _metadata = {}
    missing_counter = 0
    logger.info("Building metadata dataframe. This may take a while...")
    for scan in tqdm(tmp_dataloader, total=len(image_files)):
        meta = scan._meta
        filename = meta["filename_or_obj"][0].split("/")[-1]
        patient = int(filename.split(".")[0].split("-")[0][-3:])
        try:
            target = int(patient_df.loc[patient, target_name])
            _metadata[patient] = filename, target
        except Exception as e:
            # missing patient in metadata
            if isinstance(e, KeyError):
                missing_counter += 1
                logger.debug(
                    f"Patient '{patient}' not found in metadata. Missing counter increased to: {missing_counter} patients."
                )
                continue
            # target is not an integer
            elif isinstance(e, TypeError):
                logger.debug(
                    f"{target}. is not an integer. Skipping patient '{patient}'..."
                )
                continue

    metadata = (
        pd.DataFrame.from_dict(
            _metadata, orient="index", columns=["filename", target_name]
        )
        .rename_axis("IXI_ID")
        .dropna()
        .dropna(subset=[target_name])
    )
    return metadata


def build_chest_xray_metadata_dataframe(cfg: Configuration, split: str):
    logger.info(f"Building chest x-ray metadata dataframe for split: '{split}'...")
    IGNORE = ".DS_Store"
    scan_data = cfg.datasets.scan_data
    split_dir = os.path.join(scan_data, split)
    split_metadata = []
    classes = sorted(os.listdir(split_dir))
    classes.remove(IGNORE)
    label_encoding = 0

    for label in classes:
        class_dir = os.path.join(split_dir, label)
        class_files = os.listdir(class_dir)
        try:
            class_files.remove(IGNORE)
        except ValueError:
            pass
        for f in class_files:
            split_metadata.append((os.path.join(class_dir, f), label_encoding))
        label_encoding += 1
    return pd.DataFrame(split_metadata, columns=["image_files", "labels"])


def instantiate_image_dataset(cfg: Configuration, save_metadata=False, **kwargs):
    """
    Instantiates a MONAI image dataset given a hydra configuration. This uses the `hydra.utils.instantiate` function to instantiate the dataset from the MONAI python package.

    ## Args
    * `cfg` (`Configuration`): The configuration.
    ## Returns
    * `monai.data.Dataset`: The instantiated dataset.
    """
    dataset_cfg: DatasetConfiguration = cfg.datasets
    if dataset_cfg.extension == ".nii.gz":
        target_name = cfg.target
        scan_data = dataset_cfg.scan_data
        scan_paths = [os.path.join(scan_data, f) for f in os.listdir(scan_data)]
        filtered_scan_paths = _filter_scan_paths(
            filter_function=lambda x: x.split("/")[-1], scan_paths=scan_paths
        )
        metadata = build_ixi_metadata_dataframe(
            cfg=cfg, image_files=filtered_scan_paths
        )
        image_files = metadata["filename"].values
        labels = metadata[target_name].values
        dataset: monai.data.Dataset = instantiate(
            config=dataset_cfg.instantiate,
            image_files=[os.path.join(scan_data, f) for f in image_files],
            labels=labels,
            **kwargs,
        )
        if save_metadata:
            metadata.to_csv(
                os.path.join(
                    DEFAULT_DATA_PATH, "patients", "ixi_image_dataset_metadata.csv"
                )
            )
        return dataset

    elif dataset_cfg.extension == ".jpeg":
        train_metadata = build_chest_xray_metadata_dataframe(cfg=cfg, split="train")
        test_metadata = build_chest_xray_metadata_dataframe(cfg=cfg, split="test")
        train_dataset = monai.data.Dataset = instantiate(
            config=dataset_cfg.instantiate,
            image_files=train_metadata["image_files"].values,
            labels=train_metadata["labels"].values,
            **kwargs,
        )
        test_dataset = monai.data.Dataset = instantiate(
            config=dataset_cfg.instantiate,
            image_files=test_metadata["image_files"].values,
            labels=test_metadata["labels"].values,
            **kwargs,
        )
        if save_metadata:
            train_metadata.to_csv(
                os.path.join(
                    DEFAULT_DATA_PATH, "patients", "chest_xray_train_metadata.csv"
                )
            )
            test_metadata.to_csv(
                os.path.join(
                    DEFAULT_DATA_PATH, "patients", "chest_xray_test_metadata.csv"
                )
            )
        return train_dataset, test_dataset
    else:
        raise ValueError(
            f"Dataset extension '{dataset_cfg.extension}' not supported. Please use '.nii.gz' or '.jpeg'."
        )


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
    X = np.array(dataset.image_files)
    encoder = LabelEncoder()
    y = encoder.fit_transform(dataset.labels)
    logger.info(f"Label encoder information for target: '{cfg.target}'")
    inspect(encoder)
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
    train_val_test_split_dict["test"] = test_dataset
    if job_cfg.perform_validation:
        ## create the train/val splits
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, stratify=y_train, **job_cfg.train_test_split
        )
        val_dataset: monai.data.Dataset = instantiate(
            config=dataset_cfg.instantiate,
            image_files=X_val,
            labels=y_val,
            transform=eval_transforms,
            **kwargs,
        )
        train_val_test_split_dict["val"] = val_dataset

    ## create train and val datasets
    train_dataset: monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=X_train,
        labels=y_train,
        transform=train_transforms,
        **kwargs,
    )
    train_val_test_split_dict["train"] = train_dataset
    logger.info("Train/val/test splits created.")
    return train_val_test_split_dict
