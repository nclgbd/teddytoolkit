import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from hydra.utils import instantiate


# sklearn
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# monai
import monai
from monai.data import ImageDataset

# rtk
from rtk import *
from rtk._datasets import create_transforms
from rtk.config import *
from rtk.utils import (
    get_logger,
    hydra_instantiate,
    load_patient_dataset,
    load_patient_dataset,
    login,
)


logger = get_logger(__name__)


def load_mimic_dataset(cfg: BaseConfiguration, save_metadata=False, **kwargs):
    dataset_cfg: DatasetConfiguration = kwargs.get("dataset_cfg", None)
    if dataset_cfg is None:
        dataset_cfg = cfg.datasets

    index = dataset_cfg.index
    target = dataset_cfg.target
    preprocessing_cfg = dataset_cfg.preprocessing
    positive_class = preprocessing_cfg.get("positive_class", "Pneumonia")

    ws = login()
    # pd.read_csv(dataset_cfg.patient_data).set_index(index)
    patient_data = load_patient_dataset(ws, dataset_cfg.patient_data).set_index(index)

    # remove all of the negative class for diffusion
    if target != "class_conditioned_labels" and "diffusion" in cfg.mode:
        logger.info("Removing all negative classes...")
        class_encoding = dataset_cfg.encoding
        patient_data = patient_data[patient_data[positive_class] == 1]

    train_data = patient_data[patient_data["split"] == "train"]
    val_data = patient_data[patient_data["split"] == "validate"]
    test_data = patient_data[patient_data["split"] == "test"]

    if save_metadata:
        patient_metadata_path = os.path.join(DEFAULT_DATA_PATH, "patients")
        train_data.to_csv(
            os.path.join(
                patient_metadata_path, f"{dataset_cfg.name}_train_metadata.csv"
            )
        )

        val_data.to_csv(
            os.path.join(patient_metadata_path, f"{dataset_cfg.name}_val_metadata.csv")
        )

        test_data.to_csv(
            os.path.join(patient_metadata_path, f"{dataset_cfg.name}_test_metadata.csv")
        )

    def __build_mimic_data_split(cfg: BaseConfiguration, data: pd.DataFrame, split=""):
        dataset_cfg = cfg.datasets
        image_files = [
            os.path.join(dataset_cfg.scan_data, f) for f in data[IMAGE_KEYNAME].values
        ]
        labels = data[dataset_cfg.target].values
        transforms = create_transforms(
            cfg, use_transforms=cfg.use_transforms if split == "train" else False
        )

        dataset = hydra_instantiate(
            dataset_cfg.instantiate,
            image_files=image_files,
            labels=labels,
            transform=transforms,
        )
        return dataset

    train_dataset = __build_mimic_data_split(cfg, train_data, split="train")
    val_dataset = __build_mimic_data_split(cfg, val_data)
    test_dataset = __build_mimic_data_split(cfg, test_data)

    return [train_dataset, val_dataset, test_dataset]
