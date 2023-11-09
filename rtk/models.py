# imports
import os
from hydra.utils import instantiate
from omegaconf import DictConfig

from azureml.core import Model, Workspace

# torch imports
import torch
import torch.nn as nn

# monai
from generative.inferers import DiffusionInferer
from generative.networks.schedulers import Scheduler

# rtk
from rtk import DEFAULT_MODEL_PATH
from rtk.config import Configuration, ModelConfiguration, DiffusionModelConfiguration
from rtk.utils import get_logger, hydra_instantiate

logger = get_logger(__name__)


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
    logger.info(f"Downloading custom model '{name}'...")
    model = Model(ws, name=name, version=version)
    model_path = os.path.join(target_dir, name)
    os.makedirs(target_dir, exist_ok=True)
    location = model.download(target_dir=model_path, exist_ok=True)
    logger.info("Download complete.")
    logger.debug(f"Model location: {location}")

    return location


def instantiate_model(
    cfg: Configuration, device: torch.device = torch.device("cpu"), **kwargs
):
    """
    Instantiates a model from the given configuration.

    ## Args:
    * `model_cfg` (`ModelConfiguration`): The model configuration.
    * `device` (`torch.device`, optional): The device to instantiate the model on. Defaults to `torch.device("cpu")`.
    """
    logger.info("Instantiating model...")
    model_cfg: ModelConfiguration = cfg.models
    model_name: str = model_cfg.model._target_.split(".")[-1]
    model: nn.Module = hydra_instantiate(cfg=model_cfg.model, **kwargs)

    load_model = model_cfg.get("load_model", None)
    if load_model is not None:
        logger.info("Loading model weights...")
        from rtk.utils import login

        ws = login()
        model_path = download_model_weights(ws, **load_model)
        model.load_state_dict(torch.load(model_path))

    return model.to(device)


def instantiate_criterion(
    cfg: Configuration, device: torch.device = torch.device("cpu"), **kwargs
):
    """
    Instantiates the criterion (loss function) from a given configuration.

    ## Args:
    * `cfg` (`Configuration`): The model configuration.
    """
    logger.info("Instantiating criterion (loss function)...")
    criterion: nn.Module = hydra_instantiate(cfg=cfg.models.criterion, **kwargs)
    return criterion.to(device)


def instantiate_optimizer(cfg: Configuration, model: nn.Module, **kwargs):
    """
    Instantiates the optimizer from a given configuration.

    ## Args:
    * `cfg` (`Configuration`): The model configuration.
    * `model` (`nn.Module`): The model to optimize.
    """
    logger.info("Instantiating optimizer...")
    optimizer: torch.optim.Optimizer = hydra_instantiate(
        cfg=cfg.models.optimizer, params=model.parameters(), **kwargs
    )
    return optimizer


def instantiate_diffusion_scheduler(model_cfg: DiffusionModelConfiguration, **kwargs):
    """
    Instantiates the scheduler from a given configuration.

    ## Args:
    * `model_cfg` (`DiffusionModelConfiguration`): The model configuration.
    """
    scheduler: Scheduler = hydra_instantiate(cfg=model_cfg.scheduler, **kwargs)
    return scheduler


def instantiate_diffusion_inferer(
    model_cfg: DiffusionModelConfiguration, scheduler: Scheduler, **kwargs
):
    """
    Instantiates the inferer from a given configuration.

    ## Args:
    * `model_cfg` (`ModelConfiguration`): The model configuration.
    * `scheduler` (`Scheduler`): The scheduler to use.
    """
    inferer: DiffusionInferer = hydra_instantiate(
        cfg=model_cfg.inference, scheduler=scheduler, **kwargs
    )
    return inferer
