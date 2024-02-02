# imports
import hydra
import numpy as np
import os
import pandas as pd
from collections import Counter
from copy import deepcopy
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from random import randint
from rich import inspect
from tqdm import tqdm

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# imblean
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import RandomOverSampler

# torch
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# monai
import monai
import monai.transforms as monai_transforms
from monai.data import ImageDataset, ThreadDataLoader, CacheDataset, PersistentDataset

# rtk
from rtk import *
from rtk._datasets import create_transforms
from rtk._datasets.cxr14 import load_cxr14_dataset
from rtk._datasets.ixi import load_ixi_dataset
from rtk._datasets.pediatrics import load_pediatrics_dataset
from rtk.config import *
from rtk.utils import (
    get_logger,
    hydra_instantiate,
    yaml_to_configuration,
)

logger = get_logger(__name__)


def visualize_scan(
    iterator: iter = None,
    index: int = None,
    scan: torch.Tensor = None,
    label: torch.Tensor = None,
):
    """"""

    if scan is None or label is None:
        scan, label = next(iterator)
        index = randint(0, len(scan) - 1) if index is None else index
        scan = scan[index]
        label = label[index]

    _filename = scan._meta["filename_or_obj"].split("/")[-1]
    patient_id = _filename.split(".")[0]

    plt.title(f"Patient ID: {patient_id}; Label: {label}")
    display_scan = scan.numpy()
    display_scan = np.transpose(display_scan, (1, 2, 0))
    plt.imshow(display_scan, cmap="bone")

    return scan, label


def resample_to_value(cfg: Configuration, metadata: pd.DataFrame, **kwargs):
    """
    Resample a dataset by duplicating the data.
    """
    dataset_cfg = cfg.datasets
    preprocessing_cfg = dataset_cfg.preprocessing
    label_encoding = sorted(metadata[LABEL_KEYNAME].unique())
    sample_to_value: int = preprocessing_cfg.get(
        "sample_to_value", kwargs.get("sample_to_value", 3500)
    )
    new_metadata = deepcopy(metadata)
    for label in label_encoding:
        class_subset = metadata[metadata[LABEL_KEYNAME] == label]
        offset_size = sample_to_value - len(class_subset)
        resampled_image_files = np.random.choice(
            class_subset[IMAGE_KEYNAME].values, size=offset_size
        )
        resampled_labels = np.array([label] * offset_size)
        resampled_dict = {
            IMAGE_KEYNAME: resampled_image_files,
            LABEL_KEYNAME: resampled_labels,
        }
        new_metadata = pd.concat(
            [
                new_metadata,
                pd.DataFrame.from_dict(
                    resampled_dict,
                ),
            ],
        )

    return new_metadata


def get_images_and_classes(dataset: ImageDataset, **kwargs):
    """
    Get the images and classes from a dataset.
    """
    try:
        images = dataset.image_files
        classes = dataset.labels
    except AttributeError:
        dataset: CacheDataset = dataset
        datalist = dataset.data
        images = [d[IMAGE_KEYNAME] for d in datalist]
        classes = [d[LABEL_KEYNAME] for d in datalist]
    return images, classes


def subset_to_class(cfg: Configuration, data_df: pd.DataFrame, **kwargs):
    dataset_cfg: DatasetConfiguration = cfg.datasets
    preprocessing_cfg = dataset_cfg.preprocessing
    subset: list = preprocessing_cfg.get("subset", kwargs.get("subset", []))
    if len(subset) > 0:
        data_df = data_df[data_df[LABEL_KEYNAME].isin(subset)]
        logger.debug(f"Subset dataframe:/n{data_df.head()}")
    return data_df


def transform_labels_to_metaclass(
    df: pd.DataFrame,
    target_name: str,
    positive_class: str,
    negative_class: str = None,
    drop: bool = True,
):
    """
    Transform the labels to the metaclass.
    """

    def _transform_to_metaclass(x, positive_class: str, negative_class: str):
        """
        Transform the label to the metaclass.
        """
        if negative_class is None:
            negative_class = f"Non-{positive_class}"
        if positive_class in x:
            return positive_class
        return negative_class

    logger.info("Transforming labels to metaclass...")
    old_target_name = f"old_{target_name}"
    df[old_target_name] = df[target_name]
    df[target_name] = df[old_target_name].apply(
        _transform_to_metaclass,
        positive_class=positive_class,
        negative_class=negative_class,
    )
    logger.info("Labels transformed.\n")
    if drop:
        df.drop(columns=[old_target_name], inplace=True)

    logger.debug(f"Dataframe:\n\n{df.head()}\n")
    classes = df[target_name].value_counts().to_dict()
    logger.info(f"Labels transformed. New class counts:\n{classes}.\n")

    return df


def preprocess_dataset(
    cfg: Configuration,
    dataset: ImageDataset,
    **kwargs,
):
    preprocessing_cfg = cfg.datasets.preprocessing
    X, y = get_images_and_classes(dataset=dataset)
    data_df = pd.DataFrame({IMAGE_KEYNAME: X, LABEL_KEYNAME: y})

    # 1. subset to class
    if preprocessing_cfg.use_subset:
        data_df = subset_to_class(cfg=cfg, data_df=data_df, **kwargs)
        X, y = data_df[IMAGE_KEYNAME].values, data_df[LABEL_KEYNAME].values

    # . convert to persistent dataset for faster lookup time
    new_dataset = PersistentDataset(
        data=data_df.to_dict(orient="records"),
        transform=dataset.transform,
        cache_dir=CACHE_DIR,
    )
    return new_dataset


def create_subset(df: pd.DataFrame, target: str, labels: list = []) -> pd.DataFrame:
    """"""

    if not any(labels):
        return df
    subset_condition = df[target].apply(lambda x: any(label in x for label in labels))
    return df[subset_condition]


def instantiate_image_dataset(
    cfg: BaseConfiguration = None, save_metadata=False, return_metadata=False, **kwargs
):
    """ """
    logger.info("Instantiating image dataset...")
    dataset_cfg: DatasetConfiguration = kwargs.get(
        "dataset_cfg", cfg.datasets if cfg is not None else None
    )

    dataset_name = dataset_cfg.name
    if dataset_name == "pediatrics":
        train_dataset, test_dataset = load_pediatrics_dataset(
            cfg=cfg,
            save_metadata=save_metadata,
            return_metadata=return_metadata,
        )

    elif dataset_name == "cxr14":
        train_dataset, test_dataset = load_cxr14_dataset(
            cfg=cfg, save_metadata=save_metadata
        )

    elif dataset_name == "ixi":
        train_dataset = load_ixi_dataset(cfg, save_metadata=save_metadata)
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' is not recognized not supported. Please use ['cxr14'|'pediatrics'|'ixi']."
        )

    logger.info("Image dataset instantiated.\n")
    return train_dataset, test_dataset


def instantiate_train_val_test_datasets(
    cfg: Configuration,
    dataset: monai.data.Dataset,
    train_transforms: monai.transforms.Compose,
    eval_transforms: monai.transforms.Compose,
    **kwargs,
):
    """
    Create train/test splits for the data.
    """
    dataset_cfg: DatasetConfiguration = cfg.datasets
    preprocessing_cfg = dataset_cfg.preprocessing
    positive_class: str = preprocessing_cfg.get("positive_class", "Pneumonia (m)")
    sklearn_cfg = cfg.sklearn
    random_state = cfg.random_state

    train_val_test_split_dict = {}
    train_test_split_kwargs: dict = sklearn_cfg.model_selection.train_test_split

    if dataset_cfg.extension == ".jpeg" or ".png":
        logger.info("Creating 'validation' split...")
        X, y = get_images_and_classes(dataset=dataset)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            stratify=y,
            random_state=random_state,
            **train_test_split_kwargs,
        )
        X_train = X_train.reshape(-1)
        train_df = pd.DataFrame({IMAGE_KEYNAME: X_train, LABEL_KEYNAME: y_train})

        sampling_method = preprocessing_cfg.sampling_method
        # if isinstance(sampling_method.method["__target__"], (os.PathLike)):
        if "_dir_" in sampling_method.method.keys():
            logger.info("Loading additional generated images...")
            gen_path = sampling_method.method["_dir_"]
            target_encoding = dataset_cfg.encoding[positive_class]

            generated_image_files = [
                os.path.join(gen_path, p) for p in os.listdir(gen_path)
            ]
            generated_labels = [target_encoding] * len(generated_image_files)
            generated_df = pd.DataFrame(
                {
                    IMAGE_KEYNAME: generated_image_files,
                    LABEL_KEYNAME: generated_labels,
                }
            )

            positive_df = train_df[train_df[LABEL_KEYNAME] == 1]
            offset = abs(len(positive_df) - len(generated_df))
            sampled_generated_df: pd.DataFrame = generated_df.sample(
                offset, random_state=cfg.random_state
            )
            logger.info(
                f"Number of sampled generated images: {len(sampled_generated_df)}"
            )
            train_df = pd.concat([train_df, sampled_generated_df])

        if preprocessing_cfg.get("use_sampling", False):
            sample_to_value: int = preprocessing_cfg.sampling_method["sample_to_value"]

            X_train = train_df[IMAGE_KEYNAME].values.reshape(-1)
            y_train = train_df[LABEL_KEYNAME].values
            # train_df = pd.DataFrame({IMAGE_KEYNAME: X_train, LABEL_KEYNAME: y_train})

            sampling_method = preprocessing_cfg.sampling_method
            if "_dir_" in sampling_method.method.keys():
                from imblearn.under_sampling import RandomUnderSampler

                sampler = RandomUnderSampler(
                    random_state=random_state,
                )
                X_train, y_train = sampler.fit_resample(X_train.reshape(-1, 1), y_train)
                X_train = X_train.reshape(-1)
                # y_train = y_train
                train_df = pd.DataFrame(
                    {IMAGE_KEYNAME: X_train, LABEL_KEYNAME: y_train}
                )
            else:
                train_df = train_df.groupby(LABEL_KEYNAME).apply(
                    lambda x: x.sample(sample_to_value, replace=True)
                )
            X_train = train_df[IMAGE_KEYNAME].values
            y_train = train_df[LABEL_KEYNAME].values

        train_dataset: monai.data.Dataset = hydra.utils.instantiate(
            config=dataset_cfg.instantiate,
            image_files=X_train,
            labels=y_train,
            transform=train_transforms,
        )
        train_val_test_split_dict["train"] = train_dataset

        if kwargs.get("perform_validation", False):
            val_dataset: monai.data.Dataset = hydra.utils.instantiate(
                config=dataset_cfg.instantiate,
                image_files=X_val,
                labels=y_val,
                transform=eval_transforms,
            )
            train_val_test_split_dict["val"] = val_dataset

    elif dataset_cfg.extension == ".nii.gz":
        logger.info("Creating train/val/test splits...")

        X = np.array(dataset.image_files)
        encoder = LabelEncoder()
        y = encoder.fit_transform(dataset.labels)
        logger.info(f"Label encoder information for target: '{dataset_cfg.target}'")
        inspect(encoder)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            stratify=y,
            random_state=random_state,
            **train_test_split_kwargs,
        )
        ## create test dataset
        test_dataset: monai.data.Dataset = hydra.utils.instantiate(
            config=dataset_cfg.instantiate,
            image_files=X_test,
            labels=y_test,
            transform=eval_transforms,
            **kwargs,
        )
        train_val_test_split_dict["test"] = test_dataset
        if kwargs.get("perform_validation", False):
            ## create the train/val splits
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                stratify=y_train,
                random_state=random_state,
                **train_test_split_kwargs,
            )
            val_dataset: monai.data.Dataset = hydra.utils.instantiate(
                config=dataset_cfg.instantiate,
                image_files=X_val,
                labels=y_val,
                transform=eval_transforms,
                **kwargs,
            )
            train_val_test_split_dict["val"] = val_dataset

        ## create train and val datasets
        train_dataset: monai.data.Dataset = hydra.utils.instantiate(
            config=dataset_cfg.instantiate,
            image_files=X_train,
            labels=y_train,
            transform=train_transforms,
            **kwargs,
        )
        train_val_test_split_dict["train"] = train_dataset
        logger.info("Train/val/test splits created.")

    else:
        raise ValueError(
            f"Dataset extension '{dataset_cfg.extension}' not supported. Please use ['.nii.gz','.jpeg','.png']."
        )

    logger.info("Train/val/test splits created.")
    logger.info("Train dataset:\t{}".format(Counter(train_dataset.labels)))
    logger.info("Val dataset:\t{}".format(Counter(val_dataset.labels)))

    return train_val_test_split_dict


def prepare_validation_dataloaders(cfg: Configuration = None, **kwargs):
    logger.info("Preparing data...")
    dataset_cfg: DatasetConfiguration = kwargs.get(
        "dataset_cfg", cfg.datasets if cfg else None
    )
    use_transforms = cfg.use_transforms
    train_transform = create_transforms(cfg, use_transforms=use_transforms)
    eval_transform = create_transforms(cfg, use_transforms=False)
    dataset = instantiate_image_dataset(cfg=cfg, transform=train_transform)
    train_dataset, test_dataset = dataset[0], dataset[-1]

    # NOTE: combined datasets here
    if len(dataset_cfg.get("additional_datasets", [])) > 0:
        combined_datasets = combine_datasets(train_dataset, test_dataset, cfg=cfg)
        train_dataset, test_dataset = combined_datasets[0], combined_datasets[-1]

    # split the dataset into train/val/test
    train_val_test_split_dict = instantiate_train_val_test_datasets(
        cfg=cfg,
        dataset=train_dataset,
        train_transforms=train_transform,
        eval_transforms=eval_transform,
    )
    train_dataset, val_dataset = (
        train_val_test_split_dict["train"],
        train_val_test_split_dict["val"],
    )
    train_dataset.transform = train_transform
    test_dataset.transform = eval_transform

    train_loader: DataLoader = hydra_instantiate(
        cfg=dataset_cfg.dataloader,
        dataset=train_dataset,
        pin_memory=torch.cuda.is_available() if torch.cuda.is_available() else False,
        shuffle=True,
    )
    val_loader: DataLoader = hydra_instantiate(
        cfg=dataset_cfg.dataloader,
        dataset=val_dataset,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
    )
    test_loader: DataLoader = hydra_instantiate(
        cfg=dataset_cfg.dataloader,
        dataset=test_dataset,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
    )

    logger.info("Data prepared.\n\n")
    return train_loader, val_loader, test_loader


def convert_image_dataset(
    dataset: ImageDataset,
    transform: monai.transforms.Compose = None,
    **kwargs,
):
    """
    Transforms a image dataset to a persistent dataset.
    """
    dataset_list = []
    images, classes = get_images_and_classes(dataset)
    items = list(zip(images, classes))

    for image_file, label in items:
        dataset_list.append({IMAGE_KEYNAME: image_file, LABEL_KEYNAME: label})

    transform = dataset.transform if transform is None else transform

    new_dataset: monai.data.Dataset = PersistentDataset(
        data=dataset_list,
        transform=transform,
        cache_dir=CACHE_DIR,
        **kwargs,
    )
    return new_dataset


def combine_datasets(
    train_dataset: monai.data.Dataset,
    test_dataset: monai.data.Dataset,
    cfg: Configuration = None,
    dataset_cfg: DatasetConfiguration = None,
    dataset_configs: list = [],
    # transform: monai.transforms.Compose = None,
    **kwargs,
):
    """
    Combine a list of datasets.
    """
    logger.info("Adding additional datasets...")
    if cfg is None:
        dataset_cfg: DatasetConfiguration = dataset_cfg
    else:
        dataset_cfg: DatasetConfiguration = cfg.datasets

    preprocessing_cfg = dataset_cfg.preprocessing

    additional_datasets: dict = dataset_cfg.get(
        "additional_datasets", {"dataset_configs": []}
    )
    dataset_configs: list = (
        dataset_configs
        if any(dataset_configs)
        else additional_datasets["dataset_configs"]
    )

    c_train_dataset = deepcopy(train_dataset)
    c_test_dataset = deepcopy(test_dataset)

    encoding: int = dataset_cfg.encoding[preprocessing_cfg["positive_class"]]
    X_train, y_train = get_images_and_classes(c_train_dataset)
    # X_test, y_test = get_images_and_classes(c_test_dataset)
    train_metadata = pd.DataFrame({IMAGE_KEYNAME: X_train, LABEL_KEYNAME: y_train})
    orig_size = len(train_metadata[train_metadata[LABEL_KEYNAME] == encoding])
    logger.info(f"Original class size: {orig_size}\n")

    # num_add_datasets = len(dataset_configs)

    for additional_dataset in dataset_configs:
        additional_dataset_name = os.path.splitext(
            additional_dataset["filepath"].split("/")[-1]
        )[0]
        logger.info(f"Adding additional dataset: '{additional_dataset_name}'...")
        filepath: os.PathLike = additional_dataset["filepath"]
        marshaller: callable = globals()[additional_dataset["loader"]]
        _dataset_cfg: DatasetConfiguration = yaml_to_configuration(filepath)
        additional_dataset_splits = marshaller(
            dataset_cfg=_dataset_cfg,
            return_metadata=True,
        )
        add_train_metadata, add_test_metadata = (
            additional_dataset_splits[2],
            additional_dataset_splits[3],
        )

        # metadata preprocessing
        train_metadata_subset = add_train_metadata[
            add_train_metadata[LABEL_KEYNAME] == encoding
        ]
        if preprocessing_cfg.get("use_sampling", False):
            train_metadata_subset = train_metadata_subset.groupby(LABEL_KEYNAME).apply(
                lambda x: x.sample(orig_size, replace=True)
            )

        test_metadata_subset = add_test_metadata[
            add_test_metadata[LABEL_KEYNAME] == encoding
        ]

        train_image_files = np.hstack(
            (
                c_train_dataset.image_files,
                train_metadata_subset[IMAGE_KEYNAME],
            )
        )
        test_image_files = np.hstack(
            (
                c_test_dataset.image_files,
                test_metadata_subset[IMAGE_KEYNAME],
            )
        )
        train_labels = np.hstack(
            (c_train_dataset.labels, train_metadata_subset[LABEL_KEYNAME])
        )
        test_labels = np.hstack(
            (c_test_dataset.labels, test_metadata_subset[LABEL_KEYNAME])
        )

        c_train_dataset = hydra_instantiate(
            cfg=dataset_cfg.instantiate,
            image_files=train_image_files,
            labels=train_labels,
            transform=c_train_dataset.transform,
        )
        c_test_dataset = hydra_instantiate(
            cfg=dataset_cfg.instantiate,
            image_files=test_image_files,
            labels=test_labels,
            transform=c_test_dataset.transform,
        )
    logger.info("Additional datasets added.")
    logger.info(f"Train dataset:\t{Counter(c_train_dataset.labels)}")
    logger.info(f"Test dataset:\t{Counter(c_test_dataset.labels)}\n")

    return c_train_dataset, c_test_dataset
