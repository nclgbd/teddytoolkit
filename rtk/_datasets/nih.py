import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from hydra.utils import instantiate

from sklearn.preprocessing import MultiLabelBinarizer

# :huggingface:
from datasets import Dataset as HGFDataset

# sklearn
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# monai
import monai

# rtk
from rtk import *
from rtk._datasets import create_transforms, load_metadata
from rtk.config import *
from rtk.utils import get_logger


logger = get_logger(__name__)

MINORITY_CLASS = "Hernia"
MINORITY_CLASS_COUNT = 227
DATA_ENTRY_PATH = (
    "/home/nicoleg/workspaces/dissertation/.data/CHEST_XRAY_14/Data_Entry_2017.csv"
)
NIH_CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


def nih_get_target_counts(df: pd.DataFrame, target: str = ""):
    return Counter(",".join(df[target]).replace("|", ",").split(","))


def load_nih_dataset(
    cfg: ImageClassificationConfiguration = None,
    save_metadata=True,
    return_metadata=False,
    subset_to_positive_class=False,
    **kwargs,
):
    dataset_cfg: ImageDatasetConfiguration = kwargs.get("dataset_cfg", None)
    if dataset_cfg is None:
        dataset_cfg = cfg.datasets

    scan_path = dataset_cfg.scan_data
    target = dataset_cfg.target
    positive_class = kwargs.get("positive_class", None)
    if positive_class is None:
        preprocessing_cfg = dataset_cfg.preprocessing
        positive_class = preprocessing_cfg.get("positive_class", "Pneumonia")

    nih_metadata = load_metadata(
        dataset_cfg.index,
        dataset_cfg.patient_dataset_name,
        dataset_cfg.patient_dataset_version,
    )

    # remove all of the negative class for diffusion
    if subset_to_positive_class:
        logger.info("Removing all negative classes...")
        nih_metadata = nih_metadata[nih_metadata[positive_class] == 1]

    # train split
    with open(os.path.join(scan_path, "train_val_list.txt"), "r") as f:
        train_val_list = [idx.strip() for idx in f.readlines()]

    train_metadata = nih_metadata[nih_metadata.index.isin(train_val_list)]
    train_transforms = kwargs.get(
        "train_transforms",
        None,
    )
    if train_transforms is None:
        train_transforms = create_transforms(
            cfg,
            use_transforms=cfg.use_transforms,
        )
    train_image_files = np.array(
        [os.path.join(scan_path, filename) for filename in train_metadata.index.values]
    )
    train_labels = list(train_metadata[target].values.tolist())
    train_dataset: monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=list(train_image_files),
        labels=list(train_labels),
        transform=train_transforms,
    )

    # test split
    with open(os.path.join(scan_path, "test_list.txt"), "r") as f:
        test_list = [idx.strip() for idx in f.readlines()]

    eval_transforms = kwargs.get(
        "eval_transforms",
        None,
    )
    if eval_transforms is None:
        eval_transforms = create_transforms(
            cfg,
            use_transforms=False,
        )
    test_metadata = nih_metadata[nih_metadata.index.isin(test_list)]
    test_image_files = np.array(
        [os.path.join(scan_path, filename) for filename in test_metadata.index.values]
    )
    test_labels = test_metadata[target].values.tolist()
    test_dataset = monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=list(test_image_files),
        labels=list(test_labels),
        transform=eval_transforms,
    )
    if save_metadata:
        train_metadata.to_csv(
            os.path.join(DEFAULT_DATA_PATH, "patients", "nih_train_metadata.csv")
        )
        test_metadata.to_csv(
            os.path.join(DEFAULT_DATA_PATH, "patients", "nih_test_metadata.csv")
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
