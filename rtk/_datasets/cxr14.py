import pandas as pd
import numpy as np
from hydra.utils import instantiate
from collections import Counter


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


def chest_xray14_get_target_counts(df: pd.DataFrame, target: str = ""):
    return Counter(",".join(df[target]).replace("|", ",").split(","))


def load_cxr14_dataset(
    cfg: Configuration = None,
    save_metadata=True,
    return_metadata=False,
    **kwargs,
):
    dataset_cfg: DatasetConfiguration = kwargs.get("dataset_cfg", None)
    if dataset_cfg is None:
        dataset_cfg = cfg.datasets

    scan_path = dataset_cfg.scan_data
    target = dataset_cfg.target
    preprocessing_cfg = dataset_cfg.preprocessing
    positive_class = preprocessing_cfg.get("positive_class", "Pneumonia")

    ws = login()
    cxr_metadata = load_patient_dataset(
        ws=ws,
        patient_dataset_name=dataset_cfg.patient_data,
        patient_dataset_version=dataset_cfg.patient_data_version,
    ).set_index(dataset_cfg.index)

    # remove all of the negative class for diffusion
    if target != "class_conditioned_labels" and "diffusion" in cfg.mode:
        logger.info("Removing all negative classes...")
        class_encoding = dataset_cfg.encoding
        cxr_metadata = cxr_metadata[
            cxr_metadata[target] == class_encoding[positive_class]
        ]

    # train split
    with open(os.path.join(scan_path, "train_val_list.txt"), "r") as f:
        train_val_list = [idx.strip() for idx in f.readlines()]

    train_metadata = cxr_metadata[cxr_metadata.index.isin(train_val_list)]
    train_transforms = create_transforms(cfg, use_transforms=cfg.use_transforms)
    train_image_files = np.array(
        [os.path.join(scan_path, filename) for filename in train_metadata.index.values]
    )
    train_labels = train_metadata[target].values
    train_dataset: monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=train_image_files,
        labels=train_labels,
        transform=train_transforms,
        **kwargs,
    )

    # test split
    with open(os.path.join(scan_path, "test_list.txt"), "r") as f:
        test_list = [idx.strip() for idx in f.readlines()]

    eval_transforms = create_transforms(cfg, use_transforms=False)
    test_metadata = cxr_metadata[cxr_metadata.index.isin(test_list)]
    test_image_files = np.array(
        [os.path.join(scan_path, filename) for filename in test_metadata.index.values]
    )
    test_labels = test_metadata[target].values

    if preprocessing_cfg.get("name", "") == "icu-preprocessing":
        logger.info("Subsetting test set to 'ICU' configuration...")
        subset = preprocessing_cfg["subset"]
        dropped_labels = list(set(class_encoding.keys()) - set(subset))
        logger.info(f"Dropping labels:\n{dropped_labels}")

        test_metadata = cxr_metadata.loc[test_metadata.index]
        _icu_query = [test_metadata[column] == 1 for column in dropped_labels]

        icu_query = pd.Series(
            np.zeros(len(test_metadata), dtype=bool),
            index=test_metadata.index,
        )

        for query in _icu_query:
            icu_query = icu_query | query

        icu_test_metadata = test_metadata.drop(index=icu_query[icu_query == True].index)
        test_metadata = test_metadata.loc[icu_test_metadata.index]

        logger.info(
            f"Number in ICU test cases: {len(icu_test_metadata)}, {len(icu_test_metadata) / len(test_metadata) * 100:.4f}%"
        )
        logger.info(
            f"Number of occurrences for 'Pneumonia' class: \n{Counter(icu_test_metadata[LABEL_KEYNAME])}"
        )

    test_dataset = monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=test_image_files,
        labels=test_labels,
        transform=eval_transforms,
        **kwargs,
    )
    if save_metadata:
        train_metadata.to_csv(
            os.path.join(DEFAULT_DATA_PATH, "patients", "cxr14_train_metadata.csv")
        )
        test_metadata.to_csv(
            os.path.join(DEFAULT_DATA_PATH, "patients", "cxr14_test_metadata.csv")
        )

    return (
        (
            train_dataset,
            test_dataset,
            train_metadata,
            test_metadata,
        )
        if return_metadata
        else (train_dataset, test_dataset)
    )
