# imports
import os
from hydra.utils import instantiate
from omegaconf import DictConfig

from azureml.core import Model, Workspace

# torch imports
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# monai
from generative.inferers import DiffusionInferer
from generative.networks.schedulers import Scheduler

# rtk
from rtk import DEFAULT_MODEL_PATH
from rtk.config import (
    ImageClassificationConfiguration,
    ModelConfiguration,
    DiffusionModelConfiguration,
)
from rtk.utils import _console, get_logger, hydra_instantiate

logger = get_logger(__name__)
console = _console


def print_trainable_parameters(model: nn.Module):
    """
    Adapted from: https://huggingface.co/docs/peft/v0.6.2/en/task_guides/image_classification_lora#load-and-prepare-a-model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    console.log(
        f"Trainable params: {trainable_params} || All params: {all_param} || Trainable%: {100 * trainable_params / all_param:.2f}"
    )


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def download_model_weights(
    ws: Workspace,
    name: str,
    version: int = 1,
    target_dir: str = DEFAULT_MODEL_PATH,
    **kwargs,
):
    """
    Downloads the pretrained weights for the SwinTransformer model.

    ## Args:
        `ws` (`Workspace`): The workspace to download the model from.
        `name` (`str`): The name of the model to download.
        `target_dir` (`str`, optional): The path to save the weights. Defaults to `./assets/model_swinvit.pt`.
    """
    console.log(f"Downloading custom model '{name}'...")
    model = Model(ws, name=name, version=version)
    model_path = os.path.join(target_dir, name)
    os.makedirs(target_dir, exist_ok=True)

    location = model.download(target_dir=model_path, exist_ok=True)
    console.log("Download complete.")
    logger.debug(f"Model location: {location}")

    return location


def instantiate_model(
    cfg: ImageClassificationConfiguration,
    device: torch.device = torch.device("cpu"),
    use_huggingface=False,
    **kwargs,
):
    """
    Instantiates a model from the given configuration.

    ## Args:
    * `model_cfg` (`ModelConfiguration`): The model configuration.
    * `device` (`torch.device`, optional): The device to instantiate the model on. Defaults to `torch.device("cpu")`.
    """
    console.log("Instantiating model...")
    model_cfg: ModelConfiguration = (
        cfg.models if kwargs.get("model_cfg", None) is None else kwargs.get("model_cfg")
    )
    model: nn.Module = hydra_instantiate(cfg=model_cfg.model, **kwargs)

    if cfg.models.get("last_layer", False):
        model.op_threshs = None  # prevent pre-trained model calibration
        model.classifier = hydra_instantiate(cfg.models.last_layer)

    pretrained_weights = model_cfg.get("pretrained_weights", None)
    if pretrained_weights is not None:
        console.log("Loading model weights...")
        from rtk.utils import login

        ws = login()

        try:
            model_name = pretrained_weights.get("name", "")
            model_dir = os.path.join(DEFAULT_MODEL_PATH, model_name)
            path = os.listdir(model_dir)[0]
            model_path = os.path.join(model_dir, path)
            console.log(f"Found model at: '{model_path}'.")
        except Exception:
            model_path = download_model_weights(ws, **pretrained_weights)

        if use_huggingface:
            model = model.from_pretrained(model_path)
        else:
            model.load_state_dict(torch.load(model_path))

    if cfg.job.get("use_multi_gpu", False):
        console.log("Using multi-GPU...")
        device_ids = kwargs.get("device_ids", [device])
        model = DDP(model, device_ids=device_ids, output_device=0)
        return model

    return model.to(device)


def instantiate_criterion(
    cfg: ImageClassificationConfiguration,
    device: torch.device = torch.device("cpu"),
    **kwargs,
):
    """
    Instantiates the criterion (loss function) from a given configuration.

    ## Args:
    * `cfg` (`Configuration`): The model configuration.
    """
    console.log("Instantiating criterion (loss function)...")
    criterion: nn.Module = hydra_instantiate(cfg=cfg.models.criterion, **kwargs)
    return criterion.to(device)


def instantiate_optimizer(
    cfg: ImageClassificationConfiguration, model: nn.Module, **kwargs
):
    """
    Instantiates the optimizer from a given configuration.

    ## Args:
    * `cfg` (`Configuration`): The model configuration.
    * `model` (`nn.Module`): The model to optimize.
    """
    console.log("Instantiating optimizer...")
    optimizer: torch.optim.Optimizer = hydra_instantiate(
        cfg=cfg.models.optimizer, params=model.parameters(), **kwargs
    )
    return optimizer


def instantiate_diffusion_scheduler(cfg: DiffusionModelConfiguration, **kwargs):
    """
    Instantiates the scheduler from a given configuration.

    ## Args:
    * `model_cfg` (`DiffusionModelConfiguration`): The model configuration.
    """
    model_cfg = cfg.models
    scheduler: Scheduler = hydra_instantiate(cfg=model_cfg.scheduler, **kwargs)
    return scheduler


def instantiate_diffusion_inferer(
    cfg: ImageClassificationConfiguration, scheduler: Scheduler, **kwargs
):
    """
    Instantiates the inferer from a given configuration.

    ## Args:
    * `model_cfg` (`ModelConfiguration`): The model configuration.
    * `scheduler` (`Scheduler`): The scheduler to use.
    """
    model_cfg = cfg.models
    inferer: DiffusionInferer = hydra_instantiate(
        cfg=model_cfg.inference, scheduler=scheduler, **kwargs
    )
    return inferer
