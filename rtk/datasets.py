# imports
import logging
from typing import List
import hydra
import numpy as np
import os
import pandas as pd
from PIL import Image
from collections import Counter
from copy import deepcopy
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from random import randint
from rich import inspect

import datasets
import transformers

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# torch
import torch
from torch.utils.data import DataLoader
from torchvision.io import read_image

# :huggingface:
from transformers import AutoTokenizer

# monai
import monai
import monai.transforms as monai_transforms
from monai.data import ImageDataset, CacheDataset, PersistentDataset

# rtk
from rtk import *
from rtk._datasets import *
from rtk._datasets.ixi import *
from rtk._datasets.mimic import *
from rtk._datasets.nih import *
from rtk._datasets.pediatrics import load_pediatrics_dataset
from rtk.config import *
from rtk.utils import (
    _console,
    get_logger,
    hydra_instantiate,
)

logger = get_logger(__name__, level=logging.DEBUG)
console = _console


def visualize_scan(
    iterator: iter = None,
    index: int = None,
    scan: torch.Tensor = None,
    label: torch.Tensor = None,
    display: bool = True,
):
    """"""

    if scan is None and label is None:
        scan, label = next(iterator)
        index = randint(0, len(scan) - 1) if index is None else index
        scan = scan[index]
        label = label[index]

    try:
        _filename = scan._meta["filename_or_obj"].split("/")[-1]
        patient_id = _filename.split(".")[0]
    except AttributeError:
        patient_id = None

    plt.title(f"Patient ID: {patient_id}; Label: {label}")
    display_scan = scan.cpu().numpy()
    display_scan = np.transpose(display_scan, (1, 2, 0))
    if display:
        plt.imshow(display_scan, cmap="bone")

    return display_scan, label


def get_images_and_classes(dataset: ImageDataset, **kwargs):
    """
    Get the images and classes from a dataset.
    """
    try:
        images = list(dataset.image_files)
        classes = list(dataset.labels)
    except AttributeError:
        dataset: CacheDataset = dataset
        datalist = dataset.data
        images = [d[IMAGE_KEYNAME] for d in datalist]
        classes = [d[LABEL_KEYNAME] for d in datalist]
    return images, classes


def subset_to_class(cfg: ImageConfiguration, data_df: pd.DataFrame, **kwargs):
    dataset_cfg: ImageDatasetConfiguration = cfg.datasets
    preprocessing_cfg = dataset_cfg.preprocessing
    subset: list = preprocessing_cfg.get("subset", kwargs.get("subset", []))
    if len(subset) > 0:
        data_df = data_df[data_df[LABEL_KEYNAME].isin(subset)]
        logger.debug(f"Subset dataframe:/n{data_df.head()}")
    return data_df


def get_target_breakdown(cfg: ImageConfiguration, datasets: list):
    dataset_cfg = cfg.datasets
    train_dataset, test_dataset = datasets[0], datasets[-1]
    class_encoding = {v: k for k, v in dataset_cfg.encoding.items()}
    concats = []

    # train split
    train_dataset_counts = Counter(train_dataset.labels)
    train_target_breakdown = pd.DataFrame.from_dict(
        train_dataset_counts, orient="index", columns=["Train split"]
    )
    train_target_breakdown.index = train_target_breakdown.index.map(class_encoding)
    concats.append(train_target_breakdown)

    # validation split
    if len(datasets) == 3:
        val_dataset_counts = Counter(datasets[1].labels)
        val_target_breakdown = pd.DataFrame.from_dict(
            val_dataset_counts, orient="index", columns=["Validation split"]
        )
        val_target_breakdown.index = val_target_breakdown.index.map(class_encoding)
        concats.append(val_target_breakdown)

    # test split
    test_dataset_counts = Counter(test_dataset.labels)
    test_target_breakdown = pd.DataFrame.from_dict(
        test_dataset_counts, orient="index", columns=["Test split"]
    )
    test_target_breakdown.index = test_target_breakdown.index.map(class_encoding)
    concats.append(test_target_breakdown)

    target_breakdown = pd.concat(concats, axis=1)
    target_breakdown["Total"] = target_breakdown.sum(axis=1)
    split_totals = target_breakdown.sum(axis=0)
    split_totals.name = "Total"
    target_breakdown = pd.concat([target_breakdown, split_totals.to_frame().T], axis=0)
    return target_breakdown


def set_labels_from_encoding(cfg: ImageConfiguration, encoding: dict = None):
    dataset_cfg = cfg.datasets
    encoding = dataset_cfg.encoding if encoding is None else encoding
    if dataset_cfg.labels is not None and not any(dataset_cfg.labels):
        cfg.datasets.labels = list(encoding.keys())

    return list(encoding.keys())


def create_subset(df: pd.DataFrame, target: str, labels: list = []) -> pd.DataFrame:
    """"""

    if not any(labels):
        return df
    subset_condition = df[target].apply(lambda x: any(label in x for label in labels))
    return df[subset_condition]


def apply_label_to_text_prompts(x, base: list = None):
    if base == None:
        base = "A photo of a lung xray".split(" ")
    if x["No Finding"] == 1 or sum(x.values) == 0:
        return " ".join(base).lower()

    prompt = base.copy()
    prompt.extend(["depicting", "visible"])

    classes = []
    for i, s in enumerate(x):
        if s == 1:
            clss = x.index[i] + ","
            classes.append(clss)

    if len(classes) >= 2:
        classes.insert(-1, "and")

    prompt.extend(classes)
    prompt = " ".join(prompt).lower()

    len_prompt = len(prompt.split(","))
    if len_prompt == 3:
        prompt = prompt.replace(",", "")
        return prompt

    return prompt[:-1]  # remove the last comma


def instantiate_text_dataset(
    cfg: TextConfiguration,
    subset_to_positive_class=False,
    tokenizer: AutoTokenizer = None,
    **kwargs,
):
    dataset_cfg: DatasetConfiguration = kwargs.get("dataset_cfg", None)
    if dataset_cfg is None:
        dataset_cfg = cfg.datasets
    index = dataset_cfg.index
    data_path = dataset_cfg.scan_data
    preprocessing_cfg = dataset_cfg.preprocessing
    positive_class = kwargs.get("positive_class", None)
    if positive_class is None:
        positive_class = preprocessing_cfg.get("positive_class", "Pneumonia")

    ws: Workspace = kwargs.get("ws", None)
    metadata = load_metadata(
        index,
        ws=ws,
        patient_data_name=dataset_cfg.patient_data,
        patient_data_version=dataset_cfg.patient_data_version,
    )

    class_names = (
        NIH_CLASS_NAMES
        if dataset_cfg.name == "nih" or dataset_cfg.name == "cxr14"
        else MIMIC_CLASS_NAMES
    )

    if dataset_cfg.name == "nih" or dataset_cfg.name == "cxr14":
        from rtk._datasets.nih import DATA_ENTRY_PATH

        id2label = {i: l for i, l in enumerate(class_names)}
        label2id = {l: i for i, l in enumerate(class_names)}
        encodings = {"id2label": id2label, "label2id": label2id}

        data_entry = pd.read_csv(DATA_ENTRY_PATH).set_index("Image Index")
        finding_labels = data_entry["Finding Labels"]
        metadata = metadata.join(finding_labels, on="Image Index")
        metadata["multiclass_labels"] = metadata["Finding Labels"].apply(
            lambda x: [x_i for x_i in x.split("|")] if "|" in x else [x]
        )

        # remove all of the negative class for diffusion
        if subset_to_positive_class:
            console.log("Removing all negative classes...")
            metadata = metadata[metadata[positive_class] == 1]

        if cfg.mode == "evaluate":
            overlapped_classes = set(NIH_CLASS_NAMES).intersection(
                set(MIMIC_CLASS_NAMES)
            )
            drop_classes = list(set(class_names) - set(overlapped_classes))
            for d in drop_classes:
                metadata = metadata.drop(metadata[metadata[d] == 1].index)
            console.log(
                f"Overlapped classes: {overlapped_classes}. Dropping: {drop_classes}"
            )

        mlb = MultiLabelBinarizer(classes=NIH_CLASS_NAMES)
        mlb.fit(metadata["multiclass_labels"])

        # Create train and test splits
        console.log("Creating huggingface dataset...")

        def split_data(metadata: dict, split: str = "train"):
            console.log(f"Creating '{split}' split...")
            if split == "train":
                # train split
                with open(os.path.join(data_path, "train_val_list.txt"), "r") as f:
                    train_val_list = [idx.strip() for idx in f.readlines()]
                    metadata = metadata[metadata.index.isin(train_val_list)]
                    metadata, val_metadata = train_test_split(
                        metadata,
                        stratify=metadata[positive_class],
                        random_state=cfg.random_state,
                        **cfg.sklearn.model_selection.train_test_split,
                    )

                    if (
                        preprocessing_cfg.use_sampling
                        and subset_to_positive_class == False
                        and cfg.mode != "evaluate"
                    ):
                        metadata = resample_to_value(cfg, metadata, class_names)
                    return metadata, val_metadata
            else:
                # test split
                with open(os.path.join(data_path, "test_list.txt"), "r") as f:
                    test_list = [idx.strip() for idx in f.readlines()]
                    metadata = metadata[metadata.index.isin(test_list)]
                    return metadata

        train_metadata, val_metadata = split_data(metadata, split="train")
        test_metadata = split_data(metadata, split="test")

        for split, data in {
            "train": train_metadata,
            "validation": val_metadata,
            "test": test_metadata,
        }.items():
            class_counts = data[class_names].sum()
            console.log(f"{split.capitalize()} class counts:\n{class_counts}")

        train_dataset = create_text_dataset(
            train_metadata, tokenizer=tokenizer, split="train"
        )
        eval_dataset = create_text_dataset(
            val_metadata, tokenizer=tokenizer, split="validation"
        )
        test_dataset = create_text_dataset(
            test_metadata, tokenizer=tokenizer, split="test"
        )

        ret: list = [train_dataset, eval_dataset, test_dataset, encodings]
        return ret

    elif dataset_cfg.name == "mimic-cxr":
        ret: list = load_mimic_text_dataset(
            cfg,
            metadata,
            tokenizer=tokenizer,
            subset_to_positive_class=subset_to_positive_class,
            **kwargs,
        )
        return ret


def instantiate_image_dataset(
    cfg: ImageConfiguration = None,
    save_metadata=False,
    return_metadata=False,
    **kwargs,
) -> List[ImageDataset]:
    """ """
    console.log("Instantiating image dataset...")
    dataset_cfg: ImageDatasetConfiguration = kwargs.get(
        "dataset_cfg", cfg.datasets if cfg is not None else None
    )
    set_labels_from_encoding(cfg=cfg)

    dataset_name = dataset_cfg.name

    if dataset_name == "nih" or dataset_name == "cxr14":
        loaded_datasets = load_nih_image_dataset(
            cfg=cfg,
            return_metadata=return_metadata,
            save_metadata=save_metadata,
            **kwargs,
        )

    elif dataset_name == "mimic-cxr":
        loaded_datasets = load_mimic_image_datasets(
            cfg=cfg,
            return_metadata=return_metadata,
            save_metadata=save_metadata,
            **kwargs,
        )

    elif dataset_name == "pediatrics":
        loaded_datasets = load_pediatrics_dataset(
            cfg=cfg,
            return_metadata=return_metadata,
            save_metadata=save_metadata,
        )

    elif dataset_name == "ixi":
        loaded_datasets = load_ixi_dataset(cfg, save_metadata=save_metadata)

    else:
        raise ValueError(
            f"Dataset '{dataset_name}' is not recognized not supported. Please use ['nih'|'cxr14'|'pediatrics'|'ixi'|'mimic']."
        )

    if "additional_datasets" in dataset_cfg:
        console.log("Adding additional datasets...")
        train_dataset, test_dataset = loaded_datasets[0], loaded_datasets[-1]
        val_dataset = None if len(loaded_datasets) == 2 else loaded_datasets[1]
        combined_datasets = combine_datasets(
            cfg=cfg,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
        )
        train_dataset, test_dataset = combined_datasets[0], combined_datasets[-1]
        if len(combined_datasets) == 3:
            loaded_datasets: list = (
                train_dataset,
                combined_datasets[1],
                test_dataset,
            )
        else:
            loaded_datasets: list = train_dataset, test_dataset

    console.log("Image dataset instantiated.")
    return loaded_datasets


def combine_datasets(
    cfg: ImageConfiguration,
    train_dataset: ImageDataset,
    test_dataset: ImageDataset,
    val_dataset: ImageDataset = None,
    positive_class: str = "Pneumonia",
    **kwargs,
):
    dataset_cfg: ImageDatasetConfiguration = cfg.datasets
    dataset_configs = list(dataset_cfg.additional_datasets.dataset_configs)
    for dc in dataset_configs:
        d_cfg = OmegaConf.load(dc["filepath"])
        loader = hydra_instantiate(dc["loader"])
        full_datasets: List[ImageDataset] = loader(
            random_state=cfg.random_state,
            dataset_cfg=d_cfg,
            preprocessing_cfg=dataset_cfg.preprocessing,
            positive_class=positive_class,
            target=dataset_cfg.target,
            train_transforms=train_dataset.transform,
            eval_transforms=test_dataset.transform,
        )
        a_train_data, a_test_data = full_datasets[0], full_datasets[-1]
        train_dataset = ImageDataset(
            image_files=list(train_dataset.image_files)
            + list(a_train_data.image_files),
            labels=list(train_dataset.labels) + list(a_train_data.labels),
            transform=train_dataset.transform,
        )
        test_dataset = ImageDataset(
            image_files=list(test_dataset.image_files) + list(a_test_data.image_files),
            labels=list(test_dataset.labels) + list(a_test_data.labels),
            transform=test_dataset.transform,
        )
        if len(full_datasets) == 3:
            a_val_data = full_datasets[1]
            val_dataset = ImageDataset(
                image_files=list(val_dataset.image_files)
                + list(a_val_data.image_files),
                labels=list(val_dataset.labels) + list(a_val_data.labels),
                transform=test_dataset.transform,
            )

    return [train_dataset, val_dataset, test_dataset]


def instantiate_train_val_test_datasets(
    cfg: ImageConfiguration,
    dataset: monai.data.ImageDataset,
    train_transforms: monai.transforms.Compose,
    eval_transforms: monai.transforms.Compose,
    **kwargs,
):
    """
    Create train/test splits for the data.
    """
    dataset_cfg: ImageDatasetConfiguration = cfg.datasets
    preprocessing_cfg = dataset_cfg.preprocessing
    positive_class: str = preprocessing_cfg.get("positive_class", "Pneumonia")
    sklearn_cfg = cfg.sklearn
    random_state = cfg.random_state

    train_val_test_split_dict = {}
    train_test_split_kwargs: dict = sklearn_cfg.model_selection.train_test_split

    if dataset_cfg.extension == ".jpeg" or ".png":
        console.log("Creating 'validation' split...")
        X, y = get_images_and_classes(dataset=dataset)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            stratify=y,
            random_state=random_state,
            **train_test_split_kwargs,
        )
        train_data = pd.DataFrame({IMAGE_KEYNAME: X_train, LABEL_KEYNAME: y_train})

        sampling_method = preprocessing_cfg.sampling_method
        if "_dir_" in sampling_method.method.keys():
            console.log("Loading additional generated images...")
            gen_path: str = sampling_method.method["_dir_"]
            target_encoding = dataset_cfg.encoding[positive_class]

            gen_image_files = [os.path.join(gen_path, p) for p in os.listdir(gen_path)]
            gen_labels = [target_encoding] * len(gen_image_files)
            gen_data = pd.DataFrame(
                {
                    IMAGE_KEYNAME: gen_image_files,
                    LABEL_KEYNAME: gen_labels,
                }
            )

            pos_data = train_data[train_data[LABEL_KEYNAME] == 1]
            offset = abs(len(pos_data) - len(gen_data))
            sampled_gen_data: pd.DataFrame = gen_data.sample(
                offset, random_state=cfg.random_state
            )
            console.log(f"Number of sampled generated images: {len(sampled_gen_data)}")
            train_data = pd.concat([train_data, sampled_gen_data])

        if preprocessing_cfg.get("use_sampling", False):
            sample_to_value: int = preprocessing_cfg.sampling_method["sample_to_value"]

            X_train = train_data[IMAGE_KEYNAME].values.reshape(-1)
            y_train = train_data[LABEL_KEYNAME].values

            sampling_method = preprocessing_cfg.sampling_method
            if "_dir_" in sampling_method.method.keys():
                from imblearn.under_sampling import RandomUnderSampler

                sampler = RandomUnderSampler(
                    random_state=random_state,
                )
                X_train, y_train = sampler.fit_resample(X_train.reshape(-1, 1), y_train)
                X_train = X_train.reshape(-1)
                # y_train = y_train
                train_data = pd.DataFrame(
                    {IMAGE_KEYNAME: X_train, LABEL_KEYNAME: y_train}
                )
            else:
                train_data = train_data.groupby(LABEL_KEYNAME).apply(
                    lambda x: x.sample(sample_to_value, replace=True)
                )
            X_train = train_data[IMAGE_KEYNAME].values
            y_train = train_data[LABEL_KEYNAME].values

        train_dataset: monai.data.Dataset = hydra.utils.instantiate(
            config=dataset_cfg.instantiate,
            image_files=X_train,
            labels=y_train,
            transform=train_transforms,
        )
        train_val_test_split_dict["train"] = train_dataset

        if kwargs.get("perform_validation", True):
            val_dataset: monai.data.Dataset = hydra.utils.instantiate(
                config=dataset_cfg.instantiate,
                image_files=X_val,
                labels=y_val,
                transform=eval_transforms,
            )
            train_val_test_split_dict["val"] = val_dataset

    elif dataset_cfg.extension == ".nii.gz":
        console.log("Creating train/val/test splits...")

        X = np.array(dataset.image_files)
        encoder = LabelEncoder()
        y = encoder.fit_transform(dataset.labels)
        console.log(f"Label encoder information for target: '{dataset_cfg.target}'")
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
        console.log("Train/val/test splits created.")

    else:
        raise ValueError(
            f"Dataset extension '{dataset_cfg.extension}' not supported. Please use ['.nii.gz','.jpeg','.png']."
        )

    console.log("Train/val/test splits created.")
    console.log("Train dataset:\t{}".format(Counter(train_dataset.labels)))
    if kwargs.get("perform_validation", True):
        console.log("Val dataset:\t{}".format(Counter(val_dataset.labels)))

    return train_val_test_split_dict


def prepare_validation_dataloaders(cfg: ImageConfiguration = None, **kwargs):
    console.log("Preparing data...")
    dataset_cfg: ImageDatasetConfiguration = kwargs.get(
        "dataset_cfg", cfg.datasets if cfg else None
    )
    full_datasets = instantiate_image_dataset(cfg=cfg, save_metadata=True)
    train_dataset, val_dataset, test_dataset = full_datasets

    loaders = []
    train_loader: DataLoader = hydra_instantiate(
        cfg=dataset_cfg.dataloader,
        dataset=train_dataset,
        pin_memory=torch.cuda.is_available() if torch.cuda.is_available() else False,
        shuffle=True,
    )
    loaders.append(train_loader)

    if val_dataset is not None:
        val_loader: DataLoader = hydra_instantiate(
            cfg=dataset_cfg.dataloader,
            dataset=val_dataset,
            pin_memory=torch.cuda.is_available(),
            shuffle=True,
        )
        loaders.append(val_loader)

    test_loader: DataLoader = hydra_instantiate(
        cfg=dataset_cfg.dataloader,
        dataset=test_dataset,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
    )
    loaders.append(test_loader)

    console.log("Data prepared.\n")
    return loaders


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
    # load_image = re
    transforms_list = list(transform.transforms)
    transforms_list.insert(0, read_image)
    new_transforms = monai_transforms.Compose(transforms_list)

    new_dataset: monai.data.Dataset = CacheDataset(
        data=dataset_list,
        transform=new_transforms,
        # cache_dir=CACHE_DIR,
        **kwargs,
    )
    return new_dataset
