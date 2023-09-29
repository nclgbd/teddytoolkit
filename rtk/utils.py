"""
General utility functions. These are not specific to any deep learning framework, and therefore can be used in different contexts.
"""

# imports
import logging
import os
import pandas as pd
import yaml
import hydra
from argparse import Namespace
from colorlog import ColoredFormatter
from logging import Logger
from omegaconf import DictConfig
from rich.console import Console
from rich.logging import RichHandler

# azureml
from azureml.core import Workspace
from azureml.core.dataset import Dataset

# rtk
from rtk import DEFAULT_DATA_PATH

__all__ = [
    "_console",
    "_logger",
    "COLOR_LOGGER_FORMAT",
    "get_console",
    "get_logger",
    "login",
    "repl",
]

LOG_TIME_FORMAT = "[%X]"
COLOR_LOGGER_FORMAT: logging.Formatter = ColoredFormatter(
    fmt="%(name)s: %(message)s", datefmt=LOG_TIME_FORMAT
)


def login(
    from_config=True,
    **kwargs,
):
    """
    Login to AzureML workspace. If path is provided, will load from the specified config file.

    ## Args:
    * `from_config` (`bool`, optional): Whether to load from config file or provide the `subscription_id`. Defaults to `True`.
    * `kwargs` (`dict`): Keyword arguments for `Workspace()`.

    ## Returns:
    * `Workspace`: AzureML workspace object.
    """

    if from_config:
        ws = Workspace.from_config()

    else:
        ws = Workspace(**kwargs)

    _logger.debug("Workspace: {}".format(ws.name))
    return ws


def get_console(**kwargs) -> Console:
    """
    Gets the rich console object.

    ## Returns:
    * `Console`: Rich console object.

    """

    return kwargs.get("console", Console(**kwargs))


def get_logger(name: str = None, level: int = logging.INFO):
    """
    Function to get a logger with a `RichHandler`. Sets up the logger with a custom format and a `StreamHandler`.

    ## Args:
    * `name` (`str`): The name of the logger. Defaults to `None`.
    * `level` (`int`): The level of the logger. Defaults to `logging.INFO`.

    ## Returns:
    * `logging.Logger`: The logger.
    """

    logger: Logger = logging.getLogger(name)
    logger.setLevel(level=level)
    rich_handler = RichHandler(
        rich_tracebacks=True,
        level=level,
        log_time_format=LOG_TIME_FORMAT,
        console=get_console(),
    )
    rich_handler.setFormatter(COLOR_LOGGER_FORMAT)
    logger.addHandler(rich_handler)
    logger.propagate = False

    return logger


def load_patient_dataset(
    ws: Workspace,
    patient_dataset_name: str,
    patient_dataset_version="latest",
    data_dir: os.PathLike = DEFAULT_DATA_PATH,
    pandas_read_fn: callable = pd.read_csv,
    **kwargs,
):
    """
    Load a patient dataset from AzureML. If the dataset is not found locally, it will be downloaded from AzureML and saved to the local cache.

    ## Args:
    * `ws` (`Workspace`): The AzureML workspace.
    * `patient_dataset_name` (`str`): The name of the patient dataset.
    * `patient_dataset_version` (`str`, optional): The version of the patient dataset. Defaults to `"latest"`.
    * `data_dir` (`os.PathLike`, optional): The path to the data directory. Defaults to `DEFAULT_DATA_PATH`.
    * `pandas_read_fn` (`callable`, optional): The function to use to read the CSV file. Defaults to `pd.read_csv`.
    * `**kwargs`: Keyword arguments for `pandas_read_fn`.
    ## Returns:
    * `pd.DataFrame`: The patient dataset.
    """

    _logger.info(f"Patient dataset:\t\t'{patient_dataset_name}'")
    _patients_csv_path = os.path.join(
        data_dir, "patients", f"{patient_dataset_name}:{patient_dataset_version}.csv"
    )
    patients_csv_path = os.path.abspath(_patients_csv_path)
    try:
        _logger.debug(
            f"Attempting to load patient dataset from: '{patients_csv_path}'..."
        )
        patient_df = pandas_read_fn(patients_csv_path, **kwargs)

    except FileNotFoundError:
        _logger.warning(
            f"Patient dataset '{patient_dataset_name}' not found. Downloading from AzureML..."
        )
        patient_df: pd.DataFrame = Dataset.get_by_name(
            ws, name=patient_dataset_name, version=patient_dataset_version
        ).to_pandas_dataframe()
        os.makedirs(os.path.dirname(patients_csv_path), exist_ok=True)
        patient_df.to_csv(patients_csv_path, index=False)

    return patient_df


def load_scan_dataset(
    ws: Workspace,
    scan_dataset_name: str,
    scan_dataset_version="latest",
    data_dir: os.PathLike = DEFAULT_DATA_PATH,
    mount=True,
):
    """
    Load a scan dataset from AzureML. If the dataset is not found locally, it will be downloaded from AzureML and saved to the local cache.

    ## Args:
    * `ws` (`Workspace`): The AzureML workspace.
    * `scan_dataset_name` (`str`): The name of the scan dataset.
    * `scan_dataset_version` (`str`, optional): The version of the scan dataset. Defaults to `"latest"`.
    * `data_dir` (`os.PathLike`, optional): The path to the data directory. Defaults to `DEFAULT_DATA_PATH`.
    * `mount` (`bool`, optional): Whether to mount the dataset or download it. Defaults to `True`.
    """
    _logger.info(f"Scan dataset:\t\t'{scan_dataset_name}'\n")

    scan_dataset: Dataset = Dataset.get_by_name(
        ws, name=scan_dataset_name, version=scan_dataset_version
    )
    if mount:
        scan_mount = scan_dataset.mount()
        _logger.info(f"Mounting scan dataset to '{scan_mount.mount_point}'.")
        return scan_mount
    else:
        target_path = os.path.join(
            data_dir, "scans", f"{scan_dataset_name}:{scan_dataset_version}"
        )
        _logger.info(f"Downloading scan dataset to '{data_dir}'.")
        scan_dataset.download(target_path=target_path, overwrite=True)
        return scan_dataset


def hydra_instantiate(cfg: DictConfig, **kwargs):
    """
    Instantiates an object from a configuration.

    ## Args:
    * `cfg` (`DictConfig`): The Hydra config.
    * `**kwargs`: Keyword arguments for the object.
    ## Returns:
    * `Any`: The instantiated class.
    """
    target_class_name = cfg["_target_"].split(".")[-1]
    _logger.debug(
        "Instantiating object '{}' from configuration".format(target_class_name)
    )
    return hydra.utils.instantiate(cfg, **kwargs)


def yaml_to_namespace(yaml_file: os.PathLike):
    """
    Converts a YAML file to a `Namespace` object.
    ## Args:
    * `yaml_file` (`os.PathLike`): The path to the YAML file.

    ## Returns:
    * `Namespace`: The `Namespace` object.

    """
    with open(yaml_file, "r") as f:
        return Namespace(**yaml.safe_load(f))


def create_run_name(cfg, random_state: int, **kwargs):
    """Create a run name."""
    dataset_cfg = cfg.datasets
    preprocessing_cfg = dataset_cfg.preprocessing
    job_cfg = cfg.job
    model_cfg = cfg.models

    model_name = model_cfg.model._target_.split(".")[-1]
    run_name: str = model_cfg.model.get("model_name", model_name.lower())
    optimizer_name: str = model_cfg.optimizer._target_.split(".")[-1].lower()
    criterion_name: str = model_cfg.criterion._target_.split(".")[-1].lower()
    run_name += f",optimizer={optimizer_name},criterion={criterion_name}"
    if preprocessing_cfg.use_sampling:
        sample_to_value = preprocessing_cfg.sample_to_value
        run_name += f",sample_to_value={sample_to_value}"

    run_name += f",pretrained={str(job_cfg.use_pretrained_weights).lower()}"

    date = cfg.date
    postfix: str = cfg.get("postfix", "")
    timestamp = cfg.timestamp
    run_name = "".join(
        [run_name, f",seed={random_state}", f",{date};{timestamp}", f",{postfix}"]
    )
    return run_name


_console = get_console()
_logger = get_logger(__name__)
