import hydra
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


# monai
import monai
from monai.data import ImageDataset, ThreadDataLoader, CacheDataset, PersistentDataset

# rtk
from rtk import *
from rtk.config import *


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
    tmp_dataset = monai.data.ImageDataset(
        image_files=image_files, labels=labels, transform=None
    )
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
    dataset: monai.data.Dataset = hydra.utils.instantiate(
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
