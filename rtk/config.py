"""
Basic hydra template configurations for the `rtk` package.
"""

import os
import random
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig, ListConfig

# hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# torch
from transformers import TrainingArguments

# rtk
from rtk.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingConfiguration:
    name: str = ""
    positive_class: str = None
    labels: list = None
    sampling_method: DictConfig = field(
        default_factory=lambda: DictConfig({"sample_to_value": -1, "method": None})
    )
    subset: list = field(default_factory=lambda: [])
    use_sampling: bool = False
    use_subset: bool = False


@dataclass
class DatasetConfiguration:
    # name of the dataset
    name: str = ""
    # preprocessing configuration
    preprocessing: PreprocessingConfiguration = field(
        default_factory=PreprocessingConfiguration
    )
    # the path to the metadata of the dataset
    patient_data: str = ""
    patient_data_version: str = "latest"
    # the name of the index column in the metadata
    index: str = ""
    # the name of the target column in the metadata
    target: str = ""
    # the names for each label in alphabetical order
    labels: list = field(default_factory=lambda: [])
    #
    dataloader: DictConfig = field(
        default_factory=lambda: DictConfig({"_target_": "torch.utils.data.DataLoader"})
    )
    #
    additional_datasets: DictConfig = field(
        default_factory=lambda: DictConfig({"dataset_configs": [], "loader": None})
    )


@dataclass
class ImageDatasetConfiguration(DatasetConfiguration):
    # the kind of dataset to instantiate
    instantiate: DictConfig = field(
        default_factory=lambda: DictConfig({"_target_": "monai.data.ImageDataset"})
    )
    # transforms
    transforms: DictConfig = field(
        default_factory=lambda: DictConfig({"load": [], "train": []})
    )
    # encoding
    encoding: dict = field(default_factory=lambda: {})
    # text prompts
    text_prompts: dict = field(default_factory=lambda: {})

    # dimension to resize the images to
    dim: int = 224
    # the path to the scan of the dataset
    scan_data: str = ""
    scan_dataset_version: str = "latest"
    # the extension of the scan files
    extension: str = ".png"


@dataclass
class MLflowConfiguration:
    experiment_name: str = "Default"
    tracking_uri: str = "file:///home/nicoleg/mlruns/"
    start_run: dict = field(default_factory=lambda: {})


@dataclass
class BaseConfiguration:
    datasets: DatasetConfiguration = field(default_factory=DatasetConfiguration())
    mlflow: MLflowConfiguration = None
    experiment_name: str = "Default"
    dry_run: bool = False
    date: str = ""
    postfix: str = ""
    timestamp: str = ""
    # the mode for the run
    mode: str = "train"
    # the path to the output directory
    output_dir: str = "outputs"
    # the path to the log directory. appended to `output_dir`
    log_dir: str = "logs"
    # the gpu device to use
    device: str = "cpu"
    # the random seed for reproducibility
    random_state: int = random.randint(0, 8192)


@dataclass
class ImageConfiguration(BaseConfiguration):
    datasets: ImageDatasetConfiguration = field(
        default_factory=ImageDatasetConfiguration()
    )
    # whether to use transforms or not
    use_transforms: bool = False


@dataclass
class TextConfiguration(BaseConfiguration):
    datasets: DatasetConfiguration = field(default_factory=DatasetConfiguration())
    # the path to the output directory
    output_dir: str = "outputs"
    # the path to the log directory. appended to `output_dir`
    log_dir: str = "logs"
    # the random seed for reproducibility
    random_state: int = random.randint(0, 8192)


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
        default_factory=lambda: dict(),
    )


@dataclass
class SklearnConfiguration:
    """
    Configuration for the `sklearn` python library.

    ## Attributes:
    * `metrics` (`ListConfig`): The metrics to use from `sklearn.metrics`.
    * `model_selection` (`DictConfig`): The keyword arguments for `sklearn.model_selection` functions.
    """

    #
    metrics: ListConfig = field(default_factory=lambda: ListConfig([]))
    #
    model_selection: DictConfig = field(default_factory=lambda: DictConfig({}))


@dataclass
class JobConfiguration:

    # whether to run in dry run mode
    dry_run: bool = True
    # the number of iterations within each epoch
    epoch_length: int = None
    # the maximum number of epochs to train for
    max_epochs: int = 10
    # whether to create an additional validation split or just use a train/test split
    perform_validation: bool = True
    # whether to track meta data or not
    set_track_meta: bool = False
    # whether to use automatic mixed precision or not
    use_autocast: bool = True
    # whether to use AzureML
    use_azureml: bool = False
    # whether to use MLflow
    use_mlflow: bool = False
    # whether to use pretrained weights or not
    use_pretrained: bool = True


@dataclass
class ModelConfiguration:

    model: DictConfig = field(default_factory=lambda: DictConfig({"_target_": ""}))
    criterion: DictConfig = field(default_factory=lambda: DictConfig({"_target_": ""}))
    optimizer: DictConfig = field(default_factory=lambda: DictConfig({"_target_": ""}))


@dataclass
class DiffusionModelConfiguration(ModelConfiguration):
    scheduler: DictConfig = field(default_factory=lambda: DictConfig({"_target_": ""}))


@dataclass
class TorchMetricsConfiguration:
    """
    Configuration for the `torchmetrics` python library.

    ## Attributes:
    * `metrics` (`ListConfig`): The metrics to use from `torchmetrics`.
    """

    metrics: ListConfig = field(default_factory=lambda: ListConfig([]))
    remap: DictConfig = field(default_factory=lambda: DictConfig({}))


@dataclass
class ImageClassificationConfiguration(ImageConfiguration):

    job: JobConfiguration = field(default_factory=JobConfiguration())
    models: ModelConfiguration = field(default_factory=ModelConfiguration())

    # module specific configurations
    ignite: IgniteConfiguration = field(default_factory=lambda: IgniteConfiguration())
    mlflow: DictConfig = field(default_factory=lambda: DictConfig({}))
    sklearn: SklearnConfiguration = field(default_factory=SklearnConfiguration())


@dataclass
class DiffusionConfiguration(ImageClassificationConfiguration):
    torchmetrics: TorchMetricsConfiguration = field(
        default_factory=TorchMetricsConfiguration
    )


# TODO: reorganize this so that it is more modular. HugggingFaceConfiguration should be a dataclass that TextToImageConfiguration and
# TODO: NLPTConfiguration inherit from.
@dataclass
class HuggingFaceConfiguration:
    training_args: TrainingArguments = field(default_factory=lambda: TrainingArguments)
    pipeline: dict = field(default_factory=lambda: {})
    unet: dict = field(default_factory=lambda: {})
    scheduler: dict = field(default_factory=lambda: {})
    tokenizer: dict = field(default_factory=lambda: {})
    text_encoder: dict = field(default_factory=lambda: {})
    vae: dict = field(default_factory=lambda: {})
    lr_scheduler: dict = field(default_factory=lambda: {})


@dataclass
class TextToImageConfiguration(ImageConfiguration):
    huggingface: HuggingFaceConfiguration = field(
        default_factory=HuggingFaceConfiguration
    )
    # The scale of input perturbation. Recommended 0.1.
    input_perturbation: float = 0.1
    # Path to pretrained model or model identifier from huggingface.co/models.
    pretrained_model_name_or_path: str = ""
    # Revision of pretrained model identifier from huggingface.co/models.
    revision: str = ""
    # Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16
    variant: str = ""
    #
    image_column: str = "image_files"
    #
    caption_column: str = "labels"
    #
    max_train_samples: int = None
    #
    validation_prompts: list = field(default_factory=lambda: [])
    #
    output_dir: str = "artifacts"
    #
    cache_dir: str = None
    #
    seed: int = -1
    #
    resolution: int = -1
    #
    center_crop: bool = False
    #
    random_flip: bool = False
    #
    train_batch_size: int = -1
    #
    num_train_epochs: int = 100
    #
    max_train_steps: int = 400
    # Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
    gradient_accumulation_steps: int = 1
    # Initial learning rate (after the potential warmup period) to use.
    gradient_checkpointing: bool = True
    learning_rate: float = -1.0
    #
    scale_lr: bool = False
    #
    lr_scheduler: str = "constant"
    #
    lr_warmup_steps: int = 500
    #
    snr_gamma: float = None
    # Whether or not to use 8-bit Adam from bitsandbytes.
    use_8bit_adam: bool = True
    #
    allow_tf32: bool = True
    #
    use_ema: bool = True
    #
    non_ema_revision: str = None
    #
    dataloader_num_workers: int = -1
    #
    adam_beta1: float = 0.9
    #
    adam_beta2: float = 0.999
    #
    adam_weight_decay: float = 1e-2
    #
    adam_epsilon: float = 1e-8
    #
    max_grad_norm: float = 1.0
    #
    prediction_type: str = None
    #
    logging_dir: str = "outputs/logs"
    #
    mixed_precision: str = "fp16"
    #
    rank: int = 4
    #
    local_rank: int = -1
    #
    checkpointing_steps: int = 500
    #
    checkpoints_total_limit: int = None
    #
    resume_from_checkpoint: str = None
    #
    enable_xformers_memory_efficient_attention: bool = True
    #
    noise_offset: float = 0.0
    #
    validation_epochs: int = 5
    #
    tracker_project_name: str = "text2image-fine-tuning"


@dataclass
class NLPTConfiguration(BaseConfiguration):
    huggingface: HuggingFaceConfiguration = field(
        default_factory=HuggingFaceConfiguration
    )
    sklearn: SklearnConfiguration = field(default_factory=SklearnConfiguration)
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 3e-5
    lr_scheduler: str = "constant"
    max_train_samples: int = None
    metric_for_best_model: str = "eval_f1-score"
    num_train_epochs: int = 3
    pretrained_model_name_or_path: str = ""
    seed: int = 0
    weight_decay: float = 0.0


def set_hydra_configuration(
    config_name: str,
    BaseConfigurationInstance: BaseConfiguration,
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
    logger.info(f"Creating configuration: '{config_name}'\n")
    GlobalHydra.instance().clear()
    init_method(version_base="1.1", **init_method_kwargs)
    cfg: DictConfig = compose(config_name=config_name, **compose_kwargs)
    return BaseConfigurationInstance(**cfg)
