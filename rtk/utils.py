"""
General utility functions. These are not specific to any deep learning framework, and therefore can be used in different contexts.
"""

# imports
import logging
import os
import pandas as pd
import yaml
import hydra
from copy import deepcopy
from argparse import Namespace
from colorlog import ColoredFormatter
from logging import Logger
from omegaconf import DictConfig, OmegaConf
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


def get_params(cfg, **kwargs):
    params = dict(cfg)
    del params["datasets"]
    del params["huggingface"]
    del params["mlflow"]

    dataset_cfg = cfg.datasets
    params["dataset_name"] = dataset_cfg.name
    params["target"] = dataset_cfg.target
    params["dim"] = dataset_cfg.dim
    params["batch_size"] = dataset_cfg.dataloader.batch_size
    params["patient_data"] = dataset_cfg.patient_data
    params["patient_data_version"] = dataset_cfg.patient_data_version

    return params


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

    return kwargs.get("console", Console(record=True, **kwargs))


_console = get_console()


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
        console=_console,
    )
    rich_handler.setFormatter(COLOR_LOGGER_FORMAT)
    logger.addHandler(rich_handler)
    logger.propagate = False

    return logger


def load_patient_dataset(
    ws: Workspace,
    patient_data_name: str,
    patient_data_version="latest",
    data_dir: os.PathLike = DEFAULT_DATA_PATH,
    pandas_read_fn: callable = pd.read_csv,
    **kwargs,
) -> pd.DataFrame:
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

    _console.log(f"Patient dataset: '{patient_data_name}'")
    _patients_csv_path = os.path.join(
        data_dir, "patients", f"{patient_data_name}:{patient_data_version}.csv"
    )
    patients_csv_path = os.path.abspath(_patients_csv_path)
    patient_df: pd.DataFrame
    try:
        _logger.debug(
            f"Attempting to load patient dataset from: '{patients_csv_path}'..."
        )
        patient_df: pd.DataFrame = pandas_read_fn(patients_csv_path, **kwargs)

    except FileNotFoundError:
        _logger.warning(
            f"Patient dataset '{patient_data_name}' not found. Downloading from AzureML..."
        )
        patient_df: pd.DataFrame = Dataset.get_by_name(
            ws, name=patient_data_name, version=patient_data_version
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
    _console.log(f"Scan dataset: '{scan_dataset_name}'\n")

    scan_dataset: Dataset = Dataset.get_by_name(
        ws, name=scan_dataset_name, version=scan_dataset_version
    )
    if mount:
        scan_mount = scan_dataset.mount()
        _console.log(f"Mounting scan dataset to '{scan_mount.mount_point}'.")
        return scan_mount
    else:
        target_path = os.path.join(
            data_dir, "scans", f"{scan_dataset_name}:{scan_dataset_version}"
        )
        _console.log(f"Downloading scan dataset to '{data_dir}'.")
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


def yaml_to_configuration(file_path: str):
    cfg = OmegaConf.load(file_path)
    del cfg["defaults"]
    cfg = DictConfig(cfg)
    return cfg


def parse_args(args):
    # Handle the environment variables for distributed training.
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def strip_target(_dict: dict, lower=False):
    target_name: str = _dict["_target_"].split(".")[-1]
    if lower:
        target_name = target_name.lower()
    return target_name


def stringify_epoch(epoch: int, width: int = 5) -> str:
    z_epoch_str = str(epoch).zfill(width)
    return z_epoch_str


_logger = get_logger(__name__)
