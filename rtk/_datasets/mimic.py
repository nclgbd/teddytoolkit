import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from hydra.utils import instantiate

from torchvision.transforms import Compose

# sklearn
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# monai
import monai
import monai.transforms as monai_transforms
from monai.data import ImageDataset

# rtk
from rtk import *
from rtk._datasets import *
from rtk.config import *
from rtk.utils import (
    get_logger,
    hydra_instantiate,
    load_patient_dataset,
    load_patient_dataset,
    login,
)

MIMIC_CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

logger = get_logger(__name__)


def load_mimic_dataset(
    cfg: ImageConfiguration = None,
    save_metadata=False,
    subset_to_positive_class=False,
    **kwargs,
):
    dataset_cfg: ImageDatasetConfiguration = kwargs.get("dataset_cfg", None)
    if dataset_cfg is None:
        dataset_cfg = cfg.datasets
    index = kwargs.get("index", dataset_cfg.index)
    target = kwargs.get("target", dataset_cfg.target)

    preprocessing_cfg = kwargs.get("preprocessing_cfg", None)
    if preprocessing_cfg is None:
        preprocessing_cfg = dataset_cfg.preprocessing

    positive_class = kwargs.get("positive_class", None)
    if positive_class == None:
        positive_class = preprocessing_cfg.get(
            "positive_class", preprocessing_cfg.get("positive_class", "Pneumonia")
        )

    random_state = kwargs.get("random_state", None)
    if random_state is None:
        random_state = cfg.random_state

    ws = login()
    patient_data = load_patient_dataset(ws, dataset_cfg.patient_data).set_index(index)

    # remove all of the negative class for diffusion
    if subset_to_positive_class:
        logger.info("Removing all negative classes...")
        patient_data = patient_data[patient_data[positive_class] == 1]

    train_metadata = patient_data[patient_data["split"] == "train"]
    val_metadata = patient_data[patient_data["split"] == "validate"]
    test_metadata = patient_data[patient_data["split"] == "test"]

    if preprocessing_cfg.use_sampling and subset_to_positive_class == False:
        train_metadata = resample_to_value(
            train_metadata,
            MIMIC_CLASS_NAMES,
            dataset_cfg=dataset_cfg,
            preprocessing_cfg=preprocessing_cfg,
            sampling_strategy=target,
            random_state=random_state,
        )

    train_class_counts = get_class_counts(train_metadata, MIMIC_CLASS_NAMES)
    logger.info(f"Train class counts:\n{train_class_counts.to_markdown()}")

    val_class_counts = get_class_counts(val_metadata, MIMIC_CLASS_NAMES)
    logger.info(f"Validation class counts:\n{val_class_counts.to_markdown()}")

    test_class_counts = get_class_counts(test_metadata, MIMIC_CLASS_NAMES)
    logger.info(f"Test class counts:\n{test_class_counts.to_markdown()}")

    if save_metadata:
        patient_metadata_path = os.path.join(DEFAULT_DATA_PATH, "patients")
        train_metadata.to_csv(
            os.path.join(patient_metadata_path, f"mimic_train_metadata.csv")
        )

        val_metadata.to_csv(
            os.path.join(patient_metadata_path, f"mimic_val_metadata.csv")
        )

        test_metadata.to_csv(
            os.path.join(patient_metadata_path, f"mimic_test_metadata.csv")
        )

    def __build_mimic_data_split(
        dataset_cfg: DatasetConfiguration,
        data: pd.DataFrame,
        transforms: monai_transforms.Compose,
    ):
        # dataset_cfg = dataset_cfg.datasets
        image_files = [
            os.path.join(dataset_cfg.scan_data, f) for f in data[IMAGE_KEYNAME].values
        ]
        labels = list(data[target].values)

        dataset = hydra_instantiate(
            dataset_cfg.instantiate,
            image_files=image_files,
            labels=labels,
            transform=transforms,
        )
        return dataset

    # use transforms
    use_transforms = kwargs.get("use_transforms", True)
    train_transforms: Compose = kwargs.get(
        "train_transforms",
        create_transforms(dataset_cfg=dataset_cfg, use_transforms=use_transforms),
    )

    eval_transforms: Compose = kwargs.get(
        "eval_transforms",
        create_transforms(dataset_cfg=dataset_cfg, use_transforms=False),
    )

    # form datasets
    train_dataset = __build_mimic_data_split(
        dataset_cfg=dataset_cfg, data=train_metadata, transforms=train_transforms
    )
    val_dataset = __build_mimic_data_split(
        dataset_cfg=dataset_cfg, data=val_metadata, transforms=eval_transforms
    )
    test_dataset = __build_mimic_data_split(
        dataset_cfg=dataset_cfg, data=test_metadata, transforms=eval_transforms
    )

    return [train_dataset, val_dataset, test_dataset]
