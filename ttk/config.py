"""
Basic hydra template configurations for the `ttk` package.
"""
import os
import random
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig, ListConfig

# hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# ttk
from ttk.utils import get_logger

logger = get_logger(__name__)


@dataclass
class IgniteConfiguration:
    """
    Configuration for the `ignite` python library.

    ## Attributes:
    * `metrics` (`DictConfig`): The metrics to use.
    * `checkpoint` (`DictConfig`): The checkpoint configuration.
    * `early_stopping` (`DictConfig`): The early stopping configuration.
    * `lr_scheduler` (`DictConfig`): The learning rate scheduler configuration.
    * `use_checkpoint` (`bool`): Whether to use model checkpointing.
    * `use_early_stopping` (`bool`): Whether to use early stopping.
    * `use_lr_scheduler` (`bool`): Whether to use learning rate schedulers.
    * `use_multi_gpu` (`bool`): Whether to use multi gpu training.
    * `score_name` (`str`): The primary metric to use when evaluating the model. Defaults to `"accuracy"`.
    * `log_interval` (`int`): How often to log the evaluation metrics. Defaults to `1`.
    """

    # whether to use model checkpointing
    use_checkpoint: bool = False
    # whether to use early stopping
    use_early_stopping: bool = False
    # whether to use learning rate schedulers
    use_lr_scheduler: bool = False
    # whether to use multi gpu training
    use_multi_gpu: bool = False
    # the primary metric to use when evaluating the model
    score_name: str = "accuracy"
    # how often to log the evaluation metrics
    log_interval: int = 1

    # the congiuration objects for the above flags
    metrics: DictConfig = field(
        default_factory=lambda: DictConfig({"Accuracy": None, "Loss": None})
    )
    checkpoint: DictConfig = field(
        default_factory=lambda: DictConfig(
            {
                "_target_": "",
                "n_saved": 1,
                "score_name": "accuracy",
                "save_handler": "artifacts/checkpoints/",
            }
        )
    )
    early_stopping: DictConfig = field(
        default_factory=lambda: DictConfig(
            {
                "_target_": "",
                "patience": 10,
            }
        )
    )
    lr_scheduler: DictConfig = field(
        default_factory=lambda: DictConfig({"_target_": ""})
    )


@dataclass
class DatasetConfiguration:
    """
    Dataset configuration class.

    ## Attributes:
    * `patient_data` (`os.PathLike`): The path to the metadata of the dataset.
    * `scan_data` (`os.PathLike`, optional): The path to the scan of the dataset. Defaults to `"./data/"`.
    * `extension` (`str`, optional): The extension of the scan files. Defaults to `".nii.gz"`.
    * `instantiate` (`DictConfig`, optional): The kind of dataset to instantiate. Defaults to `DictConfig({"_target_": "monai.data.ImageDataset"})`.
    """

    # the path to the metadata of the dataset
    patient_data: os.PathLike = ""
    # the path to the scan of the dataset
    scan_data: os.PathLike = "./data/"
    # the extension of the scan files
    extension: str = ".nii.gz"
    # the kind of dataset to instantiate
    instantiate: DictConfig = field(
        default_factory=lambda: DictConfig({"_target_": "monai.data.ImageDataset"})
    )
    #
    dataloader: DictConfig = field(
        default_factory=lambda: DictConfig({"_target_": "torch.utils.data.DataLoader"})
    )


@dataclass
class JobConfiguration:
    """
    Job configuration class.

    ## Attributes:
    * `debug` (`bool`): Whether to run in debug mode.
    * `dry_run` (`bool`): Whether to run in dry run mode.
    * `random_state` (`int`): The random seed for reproducibility.
    * `use_azureml` (`bool`): Whether to use Azure ML.
    * `use_mlflow` (`bool`): Whether to use MLflow.
    """

    # whether to run in debug mode
    debug: bool = False
    # the gpu device to use
    device: str = "cpu"
    # whether to run in dry run mode
    dry_run: bool = True
    # whether to create an additional validation split or just use a train/test split
    perform_validation: bool = True
    # the random seed for reproducibility
    random_state: int = random.randint(0, 8192)
    # whether to track meta data or not
    set_track_meta: bool = False
    # whether to use Azure ML
    use_azureml: bool = False
    # whether to use MLFlow
    use_mlflow: bool = False
    #
    use_pretrained_weights: bool = True
    # whether to use transforms or not
    use_transforms: bool = False


@dataclass
class ModelConfiguration:
    """
    The model configuration class.

    ## Attributes:
    * `model` (`DictConfig`): The model configuration.
    * `criterion` (`DictConfig`): The criterion configuration.
    * `optimizer` (`DictConfig`): The optimizer configuration.
    """

    model: DictConfig = field(default_factory=lambda: DictConfig({"_target_": ""}))
    criterion: DictConfig = field(default_factory=lambda: DictConfig({"_target_": ""}))
    optimizer: DictConfig = field(default_factory=lambda: DictConfig({"_target_": ""}))


@dataclass
class DiffusionModelConfiguration(ModelConfiguration):
    """
    The diffusion model configuration class. Extends `ModelConfiguration`.

    ## Attributes:
    * `model` (`DictConfig`): The model configuration.
    * `criterion` (`DictConfig`): The criterion configuration.
    * `optimizer` (`DictConfig`): The optimizer configuration.
    * `scheduler` (`DictConfig`): The scheduler configuration.
    * `inference` (`DictConfig`): The inference configuration.
    """

    scheduler: DictConfig = field(default_factory=lambda: DictConfig({"_target_": ""}))
    inference: DictConfig = field(default_factory=lambda: DictConfig({"_target_": ""}))


@dataclass
class Configuration:
    """
    Configuration dataclass.

    ## Attributes:
    * `run` (`RunConfiguration`): The run configuration.
    * `dataset` (`DatasetConfiguration`): The dataset configuration.
    * `artifacts_path` (`str`): The path to the artifacts directory.
    * `results_path` (`str`): The path to the results directory.
    """

    index: str = ""
    target: str = ""
    date: str = ""
    timestamp: str = ""
    datasets: DatasetConfiguration = field(default_factory=DatasetConfiguration())
    ignite: IgniteConfiguration = field(default_factory=IgniteConfiguration())
    job: JobConfiguration = field(default_factory=JobConfiguration())
    models: ModelConfiguration = field(default_factory=ModelConfiguration())


def set_hydra_configuration(
    config_name: str,
    ConfigurationInstance: Configuration,
    init_method: callable = initialize_config_dir,
    init_method_kwargs: dict = {},
    **compose_kwargs,
):
    """
    Creates and returns a hydra configuration.

    ## Args:
    * `config_name` (`str`, optional): The name of the config (usually the file name without the .yaml extension).
    * `init_method` (`function`, optional): The initialization method to use. Should be either [`initialize`, `initialize_config_module`, `initialize_config_dir`].
    Defaults to `initialize_config_dir`.
    * `kwargs` (`dict`, optional): Keyword arguments for the `init_method` function.

    ## Returns:
    * `DictConfig`: The hydra configuration.
    """
    logger.info(f"Creating configuration: '{config_name}'")
    GlobalHydra.instance().clear()
    init_method(version_base="1.1", **init_method_kwargs)
    cfg: DictConfig = compose(config_name=config_name, **compose_kwargs)
    return ConfigurationInstance(**cfg)
