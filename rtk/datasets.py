# imports
import numpy as np
import os
import pandas as pd
from collections import Counter
from copy import deepcopy
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from random import randint
from rich import inspect
from tqdm import tqdm

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# imblean
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import RandomOverSampler

# torch
import torch
from torch.utils.data import DataLoader

# monai
import monai
import monai.transforms as monai_transforms
from monai.data import ImageDataset, ThreadDataLoader, CacheDataset, PersistentDataset

# rtk
from rtk import DEFAULT_CACHE_DIR, DEFAULT_DATA_PATH
from rtk.config import Configuration, JobConfiguration, DatasetConfiguration
from rtk.utils import get_logger, hydra_instantiate, yaml_to_configuration

logger = get_logger(__name__)

_CHEST_XRAY_TRAIN_DATASET_SIZE = 5232
_CHEST_XRAY_TEST_DATASET_SIZE = 624
_IXI_MRI_DATASET_SIZE = 538

_IMAGE_KEYNAME = "image_files"
_LABEL_KEYNAME = "labels"
_COLUMN_NAMES = [_IMAGE_KEYNAME, _LABEL_KEYNAME]
_CACHE_DIR = os.path.join(DEFAULT_CACHE_DIR, "tmp")


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
    * `model_config` (`TorchModelConfiguration`, optional): The model configuration. Defaults to `None`.
    * `use_transforms` (`bool`, optional): Whether or not to use the transforms. Defaults to `False`.
    * `transform_dicts` (`dict`, optional): The dictionary of transforms to use. Defaults to `None`.
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


# def transform_labels_to_one_vs_all(
#     df: pd.DataFrame,
#     target_name: str,
#     positive_class: str,
#     negative_class: str = None,
# ):
#     """
#     Transform the labels to one vs all.
#     """
#     logger.info("Tranforming labels...")
#     old_target_name = f"old_{target_name}"
#     df[old_target_name] = df[target_name]
#     negative_class = (
#         f"non-{positive_class}" if negative_class is None else negative_class
#     )

#     def _transform_to_positive_class(x, positive_class: str, negative_class: str):
#         """
#         Transform the label to the positive class.
#         """
#         return negative_class if x != positive_class else positive_class

#     df[target_name] = df[old_target_name].apply(
#         _transform_to_positive_class,
#         positive_class=positive_class,
#         negative_class=negative_class,
#     )
#     logger.info("Labels transformed.\n")
#     df.drop(columns=[old_target_name], inplace=True)
#     logger.debug(f"Dataframe:\n\n{df.head()}\n")

#     return df


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


def build_chest_xray_metadata_dataframe(
    cfg: Configuration = None, split: str = "", **kwargs
):
    logger.info(f"Building chest x-ray metadata dataframe for split: '{split}'...\n")
    dataset_cfg: DatasetConfiguration = kwargs.get(
        "dataset_cfg", cfg.datasets if cfg is not None else None
    )
    resample_value: int = dataset_cfg.get("resample_value", 1)
    IGNORE = ".DS_Store"
    scan_data = dataset_cfg.scan_data
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
    filename_matches = {_IMAGE_KEYNAME: [], index: []}
    if version == 1.0:
        for n in range(1, 13):
            label_path = os.path.join(scan_path, f"images_{n:03}", "images")

            for filename in os.listdir(label_path):
                filename_matches[index].append(filename)
                filename_matches[_IMAGE_KEYNAME].append(
                    os.path.join(label_path, filename)
                )
    elif version == 2.0:
        for filename in os.listdir(scan_path):
            if filename.endswith(".png"):
                filename_matches[index].append(filename)
                filename_matches[_IMAGE_KEYNAME].append(
                    os.path.join(scan_path, filename)
                )

    filename_matches = pd.DataFrame.from_dict(
        filename_matches, orient="columns"
    ).set_index(index)
    patient_df = patient_df.merge(filename_matches, on=index, how="inner")

    return patient_df


def chest_xray14_get_target_counts(df: pd.DataFrame, target: str = ""):
    return Counter(",".join(df[target]).replace("|", ",").split(","))


def create_subset(df: pd.DataFrame, target: str, labels: list = []) -> pd.DataFrame:
    """"""

    if not any(labels):
        return df
    subset_condition = df[target].apply(lambda x: any(label in x for label in labels))
    return df[subset_condition]


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


def load_chest_xray_dataset(
    cfg: Configuration = None, save_metadata=False, return_metadata=False, **kwargs
):
    dataset_cfg: DatasetConfiguration = kwargs.get(
        "dataset_cfg", cfg.datasets if cfg is not None else None
    )
    use_transforms = kwargs.get(
        "use_transforms", cfg.job if cfg is not None else {"use_transforms": True}
    )

    train_transforms = create_transforms(cfg, use_transforms=use_transforms)
    test_transforms = create_transforms(cfg, use_transforms=False)
    train_metadata = build_chest_xray_metadata_dataframe(
        cfg=cfg, split="train", **kwargs
    )
    test_metadata = build_chest_xray_metadata_dataframe(cfg=cfg, split="test", **kwargs)
    train_dataset = monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=train_metadata[_IMAGE_KEYNAME].values,
        labels=train_metadata[_LABEL_KEYNAME].values,
        transform=train_transforms,
    )
    # train_dataset: monai.data.Dataset = convert_image_dataset(train_dataset)
    test_dataset = monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=test_metadata[_IMAGE_KEYNAME].values,
        labels=test_metadata[_LABEL_KEYNAME].values,
        transform=test_transforms,
    )
    # test_dataset: monai.data.Dataset = convert_image_dataset(test_dataset)
    if save_metadata:
        train_metadata.to_csv(
            os.path.join(DEFAULT_DATA_PATH, "patients", "chest_xray_train_metadata.csv")
        )
        test_metadata.to_csv(
            os.path.join(DEFAULT_DATA_PATH, "patients", "chest_xray_test_metadata.csv")
        )

    return (
        (train_dataset, test_dataset, train_metadata, test_metadata)
        if return_metadata
        else (train_dataset, test_dataset)
    )


def load_chest_xray14_dataset(
    cfg: Configuration, save_metadata=False, return_metadata=False, **kwargs
):
    dataset_cfg: DatasetConfiguration = cfg.datasets
    preprocessing_cfg = dataset_cfg.preprocessing
    index = dataset_cfg.index
    target = dataset_cfg.target
    scan_path = dataset_cfg.scan_data
    labels = dataset_cfg.labels
    # label_encoding = {v: k for k, v in dataset_cfg.encoding.items()}

    metadata = build_chest_xray14_metadata_dataframe(cfg=cfg)
    # metadata = metadata[metadata[target].isin(labels)]
    # metadata = transform_labels_to_metaclass(metadata, target, dataset_cfg.encoding)
    # subset_condition = metadata[target].str.contains(labels[0]) | metadata[
    #     target
    # ].str.contains(labels[1])

    subset = dataset_cfg.preprocessing.get("subset", [])
    metadata = create_subset(metadata, target, subset)

    def _subset_chest14_labels(x):
        _ALL_LABELS = [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
        ]
        # subset = ["Atelectasis", "Edema", "Effusion", "Consolidation", "Pneumonia"]
        unaccepted_labels = list(set(_ALL_LABELS) - set(subset))
        multiclass_labels = x.split("|")
        for label in multiclass_labels:
            if label in unaccepted_labels:
                return False
        return True

    metadata = metadata[metadata[target].apply(_subset_chest14_labels)]
    counts = chest_xray14_get_target_counts(metadata, target=target)
    logger.info(
        f"Full dataset target counts:\n{counts}",
    )
    if preprocessing_cfg.get("positive_class", "") != "":
        positive_class: str = preprocessing_cfg.positive_class
        metadata = transform_labels_to_metaclass(
            metadata,
            target,
            positive_class,
        )
    # metadata = transform_labels_to_metaclass(metadata, target, dataset_cfg.encoding)
    metadata[target] = metadata[target].apply(lambda x: dataset_cfg.encoding[x])

    # train split
    with open(os.path.join(scan_path, "train_val_list.txt"), "r") as f:
        train_val_list = [idx.strip() for idx in f.readlines()]

    train_transforms = create_transforms(cfg, use_transforms=cfg.job.use_transforms)
    train_metadata = metadata[metadata.index.isin(train_val_list)]
    # train_metadata = train_metadata[
    #     train_metadata[target].apply(_subset_chest14_labels)
    # ]
    train_dataset: monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=train_metadata[_IMAGE_KEYNAME].values,
        labels=train_metadata[target].values,
        transform=train_transforms,
        **kwargs,
    )

    # test split
    with open(os.path.join(scan_path, "test_list.txt"), "r") as f:
        test_list = [idx.strip() for idx in f.readlines()]

    eval_transforms = create_transforms(cfg, use_transforms=False)
    test_metadata = metadata[metadata.index.isin(test_list)]
    # test_metadata = test_metadata[test_metadata[target].apply(_subset_chest14_labels)]
    test_dataset = monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=test_metadata[_IMAGE_KEYNAME].values,
        labels=test_metadata[target].values,
        transform=eval_transforms,
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

    return (
        (train_dataset, test_dataset, train_metadata, test_metadata)
        if return_metadata
        else (train_dataset, test_dataset)
    )


def instantiate_image_dataset(
    cfg: Configuration = None, save_metadata=False, return_metadata=False, **kwargs
):
    """
    Instantiates a MONAI image dataset given a hydra configuration. This uses the `hydra.utils.instantiate` function to instantiate the dataset from the MONAI python package.

    ## Args
    * `cfg` (`Configuration`): The configuration.
    ## Returns
    * `monai.data.Dataset`: The instantiated dataset.
    """
    logger.info("Instantiating image dataset...")
    dataset_cfg: DatasetConfiguration = kwargs.get(
        "dataset_cfg", cfg.datasets if cfg is not None else None
    )

    if dataset_cfg.extension == ".jpeg":
        train_dataset, test_dataset = load_chest_xray_dataset(
            cfg=cfg,
            save_metadata=save_metadata,
            return_metadata=return_metadata,
        )

    elif dataset_cfg.extension == ".png":
        train_dataset, test_dataset = load_chest_xray14_dataset(
            cfg=cfg, save_metadata=save_metadata
        )

    elif dataset_cfg.extension == ".nii.gz":
        train_dataset = load_ixi_dataset(cfg, save_metadata=save_metadata)
    else:
        raise ValueError(
            f"Dataset extension '{dataset_cfg.extension}' not supported. Please use '.nii.gz' or '.jpeg'."
        )

    logger.info("Image dataset instantiated.\n")
    return train_dataset, test_dataset


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
        logger.info("Creating 'validation' split...")
        X, y = get_images_and_classes(dataset=dataset)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            stratify=y,
            random_state=random_state,
            **train_test_split_kwargs,
        )
        if preprocessing_cfg.get("use_sampling", False):
            # X_train, y_train = RandomOverSampler(
            #     random_state=random_state
            # ).fit_resample(X_train.reshape(-1, 1), y_train)
            # try:
            #     sampling_strategy = {
            #         i: preprocessing_cfg.sampling_method["sample_to_value"]
            #         for i, _ in enumerate(sorted(dataset_cfg.labels))
            #     }
            #     X_train, y_train = make_imbalance(
            #         X=X_train,
            #         y=y_train,
            #         random_state=random_state,
            #         sampling_strategy=sampling_strategy,
            #         verbose=True,
            #     )
            # except ValueError:
            # Oversampling of majority class is also needed, so we use alternative oversampling method
            sample_to_value: int = preprocessing_cfg.sampling_method["sample_to_value"]

            X_train = X_train.reshape(-1)
            _train_df = pd.DataFrame({_IMAGE_KEYNAME: X_train, _LABEL_KEYNAME: y_train})
            _train_df = _train_df.groupby(_LABEL_KEYNAME).apply(
                lambda x: x.sample(sample_to_value, replace=True)
            )
            X_train = _train_df[_IMAGE_KEYNAME].values
            y_train = _train_df[_LABEL_KEYNAME].values

        train_dataset: monai.data.Dataset = instantiate(
            config=dataset_cfg.instantiate,
            image_files=X_train,
            labels=y_train,
            transform=train_transforms,
            **kwargs,
        )
        train_val_test_split_dict["train"] = train_dataset

        if cfg.job.perform_validation:
            val_dataset: monai.data.Dataset = instantiate(
                config=dataset_cfg.instantiate,
                image_files=X_val,
                labels=y_val,
                transform=eval_transforms,
                **kwargs,
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
            f"Dataset extension '{dataset_cfg.extension}' not supported. Please use ['.nii.gz','.jpeg','.png']."
        )

    logger.info("Train/val/test splits created.")
    logger.info("Train dataset:\t{}".format(Counter(train_dataset.labels)))
    logger.info("Val dataset:\t{}".format(Counter(val_dataset.labels)))
    # logger.info("Test dataset:\t{}\n\n".format(Counter(test_dataset.labels)))

    return train_val_test_split_dict


def prepare_data(cfg: Configuration = None, **kwargs):
    """Prepare the data by creating the train/val/test splits."""
    logger.info("Preparing data...")
    dataset_cfg: DatasetConfiguration = kwargs.get(
        "dataset_cfg", cfg.datasets if cfg else None
    )
    job_cfg: JobConfiguration = kwargs.get(
        "job_cfg",
        cfg.get("job", {"use_transforms": kwargs.get("use_transforms", True)}),
    )
    use_transforms = job_cfg["use_transforms"]
    train_transform = create_transforms(cfg, use_transforms=use_transforms)
    eval_transform = create_transforms(cfg, use_transforms=False)
    dataset = instantiate_image_dataset(cfg=cfg, transform=train_transform)
    train_dataset, test_dataset = dataset[0], dataset[1]

    # NOTE: combined datasets here
    if len(dataset_cfg.get("additional_datasets", [])) > 0:
        train_dataset, test_dataset = combine_datasets(
            train_dataset, test_dataset, cfg=cfg
        )

    # split the dataset into train/val/test
    train_val_test_split_dict = instantiate_train_val_test_datasets(
        cfg=cfg, dataset=train_dataset
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
        shuffle=True,
    )

    logger.info("Data prepared.\n\n")
    return train_loader, val_loader, test_loader


def convert_image_dataset(
    dataset: ImageDataset,
    transform: monai.transforms.Compose = None,
    Dataset: monai.data.Dataset = PersistentDataset,
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

    if Dataset is PersistentDataset:
        kwargs["cache_dir"] = _CACHE_DIR
    new_dataset: monai.data.Dataset = Dataset(
        data=dataset_list,
        transform=transform,
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
    train_metadata = pd.DataFrame({_IMAGE_KEYNAME: X_train, _LABEL_KEYNAME: y_train})
    orig_size = len(train_metadata[train_metadata[_LABEL_KEYNAME] == encoding])
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
            add_train_metadata[_LABEL_KEYNAME] == encoding
        ]
        if preprocessing_cfg.get("use_sampling", False):
            train_metadata_subset = train_metadata_subset.groupby(_LABEL_KEYNAME).apply(
                lambda x: x.sample(orig_size, replace=True)
            )

        test_metadata_subset = add_test_metadata[
            add_test_metadata[_LABEL_KEYNAME] == encoding
        ]

        train_image_files = np.hstack(
            (
                c_train_dataset.image_files,
                train_metadata_subset[_IMAGE_KEYNAME],
            )
        )
        test_image_files = np.hstack(
            (
                c_test_dataset.image_files,
                test_metadata_subset[_IMAGE_KEYNAME],
            )
        )
        train_labels = np.hstack(
            (c_train_dataset.labels, train_metadata_subset[_LABEL_KEYNAME])
        )
        test_labels = np.hstack(
            (c_test_dataset.labels, test_metadata_subset[_LABEL_KEYNAME])
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
