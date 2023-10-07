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
from sklearn.pipeline import Pipeline

# torch
import torch
from torch.utils.data import Subset

# monai
import monai
import monai.transforms as monai_transforms
from monai.data import ImageDataset, ThreadDataLoader, CacheDataset, PersistentDataset

# rtk
from rtk import DEFAULT_CACHE_DIR, DEFAULT_DATA_PATH
from rtk.config import Configuration, JobConfiguration, DatasetConfiguration
from rtk.utils import get_logger, hydra_instantiate

logger = get_logger(__name__)

_CHEST_XRAY_TRAIN_DATASET_SIZE = 5232
_CHEST_XRAY_TEST_DATASET_SIZE = 624
_IXI_MRI_DATASET_SIZE = 538

_IMAGE_KEYNAME = "image"
_LABEL_KEYNAME = "label"
_COLUMN_NAMES = [_IMAGE_KEYNAME, _LABEL_KEYNAME]
_CACHE_DIR = os.path.join(DEFAULT_CACHE_DIR, "tmp")


def create_transforms(
    cfg: Configuration,
    dataset_cfg: DatasetConfiguration = None,
    use_transforms: bool = None,
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
    logger.info("Creating transforms...\n")
    dataset_cfg = cfg.datasets if dataset_cfg is None else dataset_cfg
    use_transforms = (
        use_transforms
        if use_transforms is not None
        else cfg.job.get("use_transforms", False)
    )
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

        # diffusion transforms
        if kwargs.get("mode", "") == "diffusion":
            if use_transforms:
                rand_lambda_transform = monai_transforms.RandLambdad(
                    keys=[_LABEL_KEYNAME],
                    prob=0.15,
                    func=lambda x: -1 * torch.ones_like(x),
                )
                _ret_transforms.append(rand_lambda_transform)
            lambda_transform = monai_transforms.Lambdad(
                keys=[_LABEL_KEYNAME],
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


def resample_to_value(cfg: Configuration, metadata: pd.DataFrame, **kwargs):
    """
    Resample a dataset by duplicating the data.
    """
    dataset_cfg = cfg.datasets
    preprocessing_cfg = dataset_cfg.preprocessing
    label_encoding = sorted(metadata[_LABEL_KEYNAME].unique())
    sample_to_value: int = preprocessing_cfg.get(
        "sample_to_value", kwargs.get("sample_to_value", 3500)
    )
    new_metadata = deepcopy(metadata)
    for label in label_encoding:
        class_subset = metadata[metadata[_LABEL_KEYNAME] == label]
        offset_size = sample_to_value - len(class_subset)
        resampled_image_files = np.random.choice(
            class_subset[_IMAGE_KEYNAME].values, size=offset_size
        )
        resampled_labels = np.array([label] * offset_size)
        resampled_dict = {
            _IMAGE_KEYNAME: resampled_image_files,
            _LABEL_KEYNAME: resampled_labels,
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
        images = [d[_IMAGE_KEYNAME] for d in datalist]
        classes = [d[_LABEL_KEYNAME] for d in datalist]
    return images, classes


def subset_to_class(cfg: Configuration, data_df: pd.DataFrame, **kwargs):
    dataset_cfg: DatasetConfiguration = cfg.datasets
    preprocessing_cfg = dataset_cfg.preprocessing
    subset: list = preprocessing_cfg.get("subset", kwargs.get("subset", []))
    if len(subset) > 0:
        data_df = data_df[data_df[_LABEL_KEYNAME].isin(subset)]
        logger.debug(f"Subset dataframe:/n{data_df.head()}")
    return data_df


def preprocess_dataset(
    cfg: Configuration,
    dataset: ImageDataset,
    **kwargs,
):
    preprocessing_cfg = cfg.datasets.preprocessing
    X, y = get_images_and_classes(dataset=dataset)
    data_df = pd.DataFrame({_IMAGE_KEYNAME: X, _LABEL_KEYNAME: y})

    # 1. subset to class
    if preprocessing_cfg.use_subset:
        data_df = subset_to_class(cfg=cfg, data_df=data_df, **kwargs)
        X, y = data_df[_IMAGE_KEYNAME].values, data_df[_LABEL_KEYNAME].values

    # . convert to persistent dataset for faster lookup time
    new_dataset = PersistentDataset(
        data=data_df.to_dict(orient="records"),
        transform=dataset.transform,
        cache_dir=_CACHE_DIR,
    )
    return new_dataset


# TODO: pd.Series are saved in the dataframe for some reason, so we need to convert them to ints
def build_ixi_metadata_dataframe(
    cfg: Configuration, image_files: list, labels: list = None
):
    dataset_cfg: DatasetConfiguration = cfg.datasets
    index_name = dataset_cfg.index
    target_name = dataset_cfg.target
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
    logger.info(f"Building chest x-ray metadata dataframe for split: '{split}'...\n")
    resample_value: int = cfg.datasets.get("resample_value", 1)
    IGNORE = ".DS_Store"
    scan_data = cfg.datasets.scan_data
    split_dir = os.path.join(scan_data, split)
    split_metadata = []
    classes = sorted(os.listdir(split_dir))
    try:
        classes.remove(IGNORE)
    except ValueError:
        pass
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

        if resample_value > 1 and split == "train" and label_encoding == 0:
            original_metadata = deepcopy(split_metadata)
            for _ in range(resample_value - 1):
                split_metadata += original_metadata
        label_encoding += 1

    return pd.DataFrame(split_metadata, columns=_COLUMN_NAMES)


def build_chest_xray14_metadata_dataframe(cfg: Configuration, version: float = 2.0):
    dataset_cfg: DatasetConfiguration = cfg.datasets
    index = dataset_cfg.index
    patient_path = dataset_cfg.patient_data
    scan_path = dataset_cfg.scan_data

    patient_df = pd.read_csv(patient_path).set_index(index)
    filename_matches = {"image_files": [], index: []}
    if version == 1.0:
        for n in range(1, 13):
            label_path = os.path.join(scan_path, f"images_{n:03}", "images")

            for filename in os.listdir(label_path):
                filename_matches[index].append(filename)
                filename_matches["image_files"].append(
                    os.path.join(label_path, filename)
                )
    elif version == 2.0:
        for filename in os.listdir(scan_path):
            if filename.endswith(".png"):
                filename_matches[index].append(filename)
                filename_matches["image_files"].append(
                    os.path.join(scan_path, filename)
                )

    filename_matches = pd.DataFrame.from_dict(
        filename_matches, orient="columns"
    ).set_index(index)
    patient_df = patient_df.merge(filename_matches, on=index, how="inner")

    return patient_df


def load_ixi_dataset(cfg: Configuration, save_metadata=False, **kwargs):
    dataset_cfg: DatasetConfiguration = cfg.datasets
    target_name = dataset_cfg.target
    scan_data = dataset_cfg.scan_data
    scan_paths = [os.path.join(scan_data, f) for f in os.listdir(scan_data)]
    filtered_scan_paths = _filter_scan_paths(
        filter_function=lambda x: x.split("/")[-1], scan_paths=scan_paths
    )
    metadata = build_ixi_metadata_dataframe(cfg=cfg, image_files=filtered_scan_paths)
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


def instantiate_image_dataset(cfg: Configuration, save_metadata=False, **kwargs):
    """
    Instantiates a MONAI image dataset given a hydra configuration. This uses the `hydra.utils.instantiate` function to instantiate the dataset from the MONAI python package.

    ## Args
    * `cfg` (`Configuration`): The configuration.
    ## Returns
    * `monai.data.Dataset`: The instantiated dataset.
    """
    logger.info("Instantiating image dataset...\n")
    dataset_cfg: DatasetConfiguration = cfg.datasets

    if dataset_cfg.extension == ".jpeg":
        train_metadata = build_chest_xray_metadata_dataframe(cfg=cfg, split="train")
        test_metadata = build_chest_xray_metadata_dataframe(cfg=cfg, split="test")
        train_dataset = monai.data.Dataset = instantiate(
            config=dataset_cfg.instantiate,
            image_files=train_metadata[_IMAGE_KEYNAME].values,
            labels=train_metadata[_LABEL_KEYNAME].values,
            **kwargs,
        )
        # train_dataset: monai.data.Dataset = convert_image_dataset(train_dataset)
        test_dataset = monai.data.Dataset = instantiate(
            config=dataset_cfg.instantiate,
            image_files=test_metadata[_IMAGE_KEYNAME].values,
            labels=test_metadata[_LABEL_KEYNAME].values,
            **kwargs,
        )
        # test_dataset: monai.data.Dataset = convert_image_dataset(test_dataset)
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

    elif dataset_cfg.extension == ".png":
        index = dataset_cfg.index
        target = dataset_cfg.target
        scan_path = dataset_cfg.scan_data
        labels = dataset_cfg.labels
        # label_encoding = {v: k for k, v in dataset_cfg.encoding.items()}

        metadata = build_chest_xray14_metadata_dataframe(cfg=cfg)
        metadata = metadata[metadata[target].isin(labels)]
        metadata[target] = LabelEncoder().fit_transform(metadata[target].values)
        # train split
        with open(os.path.join(scan_path, "train_val_list.txt"), "r") as f:
            train_val_list = [idx.strip() for idx in f.readlines()]
        train_metadata = metadata[metadata.index.isin(train_val_list)]
        train_dataset: monai.data.Dataset = instantiate(
            config=dataset_cfg.instantiate,
            image_files=train_metadata["image_files"].values,
            labels=train_metadata[target].values,
            **kwargs,
        )

        # test split
        with open(os.path.join(scan_path, "test_list.txt"), "r") as f:
            test_list = [idx.strip() for idx in f.readlines()]
        test_metadata = metadata[metadata.index.isin(test_list)]
        test_dataset = monai.data.Dataset = instantiate(
            config=dataset_cfg.instantiate,
            image_files=test_metadata["image_files"].values,
            labels=test_metadata[target].values,
            **kwargs,
        )
        if save_metadata:
            train_metadata.to_csv(
                os.path.join(
                    DEFAULT_DATA_PATH, "patients", "chest_xray14_train_metadata.csv"
                )
            )
            test_metadata.to_csv(
                os.path.join(
                    DEFAULT_DATA_PATH, "patients", "chest_xray14_test_metadata.csv"
                )
            )
        return train_dataset, test_dataset

    elif dataset_cfg.extension == ".nii.gz":
        train_dataset = load_ixi_dataset(cfg, save_metadata=False)
    else:
        raise ValueError(
            f"Dataset extension '{dataset_cfg.extension}' not supported. Please use '.nii.gz' or '.jpeg'."
        )


def instantiate_train_val_test_datasets(
    cfg: Configuration, dataset: monai.data.Dataset, **kwargs
):
    """
    Create train/test splits for the data.
    """
    dataset_cfg: DatasetConfiguration = cfg.datasets
    preprocessing_cfg = dataset_cfg.preprocessing
    job_cfg: JobConfiguration = cfg.job
    random_state = job_cfg.get("random_state", kwargs.get("random_state", 0))
    use_transforms = job_cfg.use_transforms
    sklearn_cfg = cfg.sklearn

    train_val_test_split_dict = {}
    train_test_split_kwargs: dict = sklearn_cfg.model_selection.train_test_split
    train_transforms = create_transforms(cfg, use_transforms=use_transforms)
    eval_transforms = create_transforms(cfg, use_transforms=False)

    if dataset_cfg.extension == ".jpeg" or ".png":
        logger.info("Creating train/val splits...\n")
        X, y = get_images_and_classes(dataset=dataset)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            stratify=y,
            random_state=random_state,
            **train_test_split_kwargs,
        )
        if preprocessing_cfg.get("use_sampling", False):
            train_metadata = pd.DataFrame.from_dict(
                {_IMAGE_KEYNAME: X_train, _LABEL_KEYNAME: y_train}, orient="columns"
            )
            train_metadata = resample_to_value(cfg=cfg, metadata=train_metadata)
            X_train = train_metadata[_IMAGE_KEYNAME].values
            y_train = train_metadata[_LABEL_KEYNAME].values

        # train_data = list(
        #     {_IMAGE_KEYNAME: image_file, _LABEL_KEYNAME: label}
        #     for image_file, label in zip(X_train, y_train)
        # )
        train_dataset: monai.data.Dataset = instantiate(
            config=dataset_cfg.instantiate,
            image_files=X_train,
            labels=y_train,
            transform=train_transforms,
            **kwargs,
        )
        train_val_test_split_dict["train"] = train_dataset

        if cfg.job.perform_validation:
            # val_data = list(
            #     {_IMAGE_KEYNAME: image_file, _LABEL_KEYNAME: label}
            #     for image_file, label in zip(X_val, y_val)
            # )
            val_dataset: monai.data.Dataset = instantiate(
                config=dataset_cfg.instantiate,
                image_files=X_val,
                labels=y_val,
                transform=eval_transforms,
                **kwargs,
            )
            train_val_test_split_dict["val"] = val_dataset

    elif dataset_cfg.extension == ".nii.gz":
        logger.info("Creating train/val/test splits...\n")

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
                X_train,
                y_train,
                stratify=y_train,
                random_state=random_state,
                **train_test_split_kwargs,
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

    else:
        raise ValueError(
            f"Dataset extension '{dataset_cfg.extension}' not supported. Please use '.nii.gz' or '.jpeg'."
        )

    return train_val_test_split_dict


def convert_image_dataset(
    dataset: ImageDataset,
    transform: monai.transforms.Compose = None,
    cache_dir: str = _CACHE_DIR,
    **kwargs,
):
    """
    Transforms a image dataset to a persistent dataset.
    """
    dataset_list = []
    images, classes = get_images_and_classes(dataset)
    items = list(zip(images, classes))

    for image_file, label in items:
        dataset_list.append({_IMAGE_KEYNAME: image_file, _LABEL_KEYNAME: label})
        # dataset_list.append((image_file, label))

    transform = dataset.transform if transform is None else transform
    new_dataset: monai.data.Dataset = PersistentDataset(
        data=dataset_list,
        transform=transform,
        cache_dir=cache_dir,
        **kwargs,
    )
    return new_dataset
