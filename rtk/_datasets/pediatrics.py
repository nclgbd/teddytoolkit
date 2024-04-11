import hydra
import numpy as np
import pandas as pd
from collections import Counter
from copy import deepcopy
from hydra.utils import instantiate


# sklearn
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# monai
import monai

# rtk
from rtk import *
from rtk._datasets import create_transforms
from rtk.config import *
from rtk.utils import login, get_logger, load_patient_dataset


logger = get_logger(__name__)


def build_chest_xray_metadata_dataframe(
    cfg: ImageClassificationConfiguration = None, split: str = "", **kwargs
):
    logger.info(f"Building chest x-ray metadata dataframe for split: '{split}'...\n")
    dataset_cfg: ImageDatasetConfiguration = kwargs.get(
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

    return pd.DataFrame(split_metadata, columns=COLUMN_NAMES)


def load_pediatrics_dataset(
    cfg: ImageClassificationConfiguration = None, save_metadata=False, return_metadata=False, **kwargs
):
    dataset_cfg: ImageDatasetConfiguration = kwargs.get(
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
    train_dataset = monai.data.Dataset = hydra.utils.instantiate(
        config=dataset_cfg.instantiate,
        image_files=train_metadata[IMAGE_KEYNAME].values,
        labels=train_metadata[LABEL_KEYNAME].values,
        transform=train_transforms,
    )
    # train_dataset: monai.data.Dataset = convert_image_dataset(train_dataset)
    test_dataset = monai.data.Dataset = hydra.utils.instantiate(
        config=dataset_cfg.instantiate,
        image_files=test_metadata[IMAGE_KEYNAME].values,
        labels=test_metadata[LABEL_KEYNAME].values,
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
