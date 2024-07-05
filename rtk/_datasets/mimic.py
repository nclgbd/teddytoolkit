from typing import List
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
import rtk
from rtk import *
from rtk._datasets import *
from rtk._datasets.nih import NIH_CLASS_NAMES
from rtk.config import *
from rtk.utils import (
    _console,
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
console = _console


def build_patient_metadata(cfg: ImageConfiguration, **kwargs):
    scan_data = cfg.datasets.scan_data

    # load in mapping file
    df_records = pd.read_csv(
        os.path.join(scan_data, "cxr-record-list.csv.gz"), header=0, sep=","
    )

    n = df_records.shape[0]
    console.print(f"{n} DICOMs in 'MIMIC-CXR v2.0.0.'")

    n = df_records["study_id"].nunique()
    console.print(f"  {n} studies.")

    n = df_records["subject_id"].nunique()
    console.print(f"  {n} subjects.")

    df_split = pd.read_csv(os.path.join(scan_data, "mimic-cxr-2.0.0-split.csv.gz"))
    df_split.info()
    df_metadata = pd.read_csv(
        os.path.join(scan_data, "mimic-cxr-2.0.0-metadata.csv.gz")
    )
    df_metadata.info()
    drop_columns = [
        "subject_id",
        "study_id",
        "PerformedProcedureStepDescription",
        "Rows",
        "Columns",
        "StudyDate",
        "StudyTime",
        "ProcedureCodeSequence_CodeMeaning",
        "ViewCodeSequence_CodeMeaning",
        "PatientOrientationCodeSequence_CodeMeaning",
    ]
    patient_df = pd.merge(
        df_records, df_metadata.drop(columns=drop_columns), on="dicom_id"
    ).set_index("dicom_id")
    return patient_df


def load_mimic_text_dataset(
    cfg: BaseConfiguration,
    metadata: pd.DataFrame,
    tokenizer: AutoTokenizer,
    subset_to_positive_class=False,
    return_metadata=False,
    **kwargs,
):
    class_names = MIMIC_CLASS_NAMES
    random_state = cfg.random_state
    dataset_cfg = cfg.datasets
    target = dataset_cfg.target
    preprocessing_cfg = cfg.datasets.preprocessing
    positive_class = kwargs.get("positive_class", preprocessing_cfg.positive_class)
    id2label = {i: l for i, l in enumerate(class_names)}
    label2id = {l: i for i, l in enumerate(class_names)}
    encodings = {"id2label": id2label, "label2id": label2id}

    # remove all of the negative class for diffusion
    if subset_to_positive_class:
        console.log("Removing all negative classes...")
        metadata = metadata[metadata[positive_class] == 1]

    if cfg.mode == "evaluate":
        overlapped_classes = set(NIH_CLASS_NAMES).intersection(set(MIMIC_CLASS_NAMES))
        drop_classes = list(set(class_names) - set(overlapped_classes))
        for d in drop_classes:
            metadata = metadata.drop(metadata[metadata[d] == 1].index)
        logger.debug(
            f"Overlapped classes: {overlapped_classes}. Dropping: {drop_classes}"
        )
        class_counts = metadata[class_names].sum()
        console.log(f"Class counts:\n{class_counts}")

    def _create_multiclass_labels(x):
        finding_labels = []
        for column in class_names:
            if x[column] == 1:
                finding_labels.append(column)
        if finding_labels == []:
            finding_labels.append("No Finding")
        return finding_labels

    console.log("Creating multiclass labels...")
    metadata["multiclass_labels"] = metadata.apply(_create_multiclass_labels, axis=1)
    mlb = MultiLabelBinarizer(classes=class_names)
    mlb.fit(metadata["multiclass_labels"])

    # split the dataset into train/val/test
    train_metadata = metadata[metadata["split"] == "train"]
    val_metadata = metadata[metadata["split"] == "validate"]
    test_metadata = metadata[metadata["split"] == "test"]

    # resample train data if needed
    if (
        preprocessing_cfg.use_sampling
        and subset_to_positive_class == False
        and cfg.mode != "evaluate"
    ):
        train_metadata = resample_to_value(
            train_metadata,
            class_names,
            dataset_cfg=dataset_cfg,
            preprocessing_cfg=preprocessing_cfg,
            sampling_strategy=target,
            random_state=random_state,
        )

    for split, data in {
        "train": train_metadata,
        "validation": val_metadata,
        "test": test_metadata,
    }.items():
        class_counts = data[class_names].sum()
        console.log(f"'{split.capitalize()}' class counts:\n{class_counts}")

    if return_metadata:
        return train_metadata, val_metadata, test_metadata

    train_dataset: HGFDataset = create_text_dataset(
        train_metadata,
        data_path=dataset_cfg.scan_data,
        target=target,
        mlb=mlb,
        tokenizer=tokenizer,
        split="train",
        **kwargs,
    )
    eval_dataset: HGFDataset = create_text_dataset(
        val_metadata,
        target=target,
        mlb=mlb,
        data_path=dataset_cfg.scan_data,
        tokenizer=tokenizer,
        split="validation",
        **kwargs,
    )
    test_dataset: HGFDataset = create_text_dataset(
        test_metadata,
        target=target,
        mlb=mlb,
        data_path=dataset_cfg.scan_data,
        tokenizer=tokenizer,
        split="test",
        **kwargs,
    )

    ret: list = [train_dataset, eval_dataset, test_dataset, encodings]
    return ret


def load_mimic_image_datasets(
    cfg: ImageConfiguration = None,
    return_metadata=False,
    save_metadata=False,
    subset_to_positive_class=False,
    **kwargs,
) -> List[ImageDataset]:

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
        console.log("Removing all negative classes...")
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
    console.log(f"Train class counts:\n{train_class_counts}")

    val_class_counts = get_class_counts(val_metadata, MIMIC_CLASS_NAMES)
    console.log(f"Validation class counts:\n{val_class_counts}")

    test_class_counts = get_class_counts(test_metadata, MIMIC_CLASS_NAMES)
    console.log(f"Test class counts:\n{test_class_counts}")

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

    if return_metadata:
        return train_metadata, val_metadata, test_metadata

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
    # use_transforms = kwargs.get("use_transforms", True)
    train_transforms = kwargs.get(
        "train_transforms",
        None,
    )
    if train_transforms is None:
        train_transforms = create_transforms(
            cfg,
            use_transforms=cfg.use_transforms,
        )

    eval_transforms = kwargs.get(
        "eval_transforms",
        None,
    )
    if eval_transforms is None:
        eval_transforms = create_transforms(
            cfg,
            use_transforms=False,
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
