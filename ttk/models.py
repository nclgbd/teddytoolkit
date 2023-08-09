# imports
from hydra.utils import instantiate
from omegaconf import DictConfig

# torch imports
import torch
import torch.nn as nn

# monai
from generative.inferers import DiffusionInferer
from generative.networks.schedulers import Scheduler

# ttk imports
from ttk.config import ModelConfiguration, DiffusionModelConfiguration
from ttk.utils import get_logger, hydra_instantiate

logger = get_logger(__name__)


def instantiate_model(
    model_cfg: ModelConfiguration, device: torch.device = torch.device("cpu"), **kwargs
):
    """
    Instantiates a model from the given configuration.

    ## Args:
    * `model_cfg` (`ModelConfiguration`): The model configuration.
    * `device` (`torch.device`, optional): The device to instantiate the model on. Defaults to `torch.device("cpu")`.
    """
    model: nn.Module = hydra_instantiate(cfg=model_cfg.model, **kwargs)
    return model.to(device)


def instantiate_criterion(
    model_cfg: ModelConfiguration, device: torch.device = torch.device("cpu"), **kwargs
):
    """
    Instantiates the criterion (loss function) from a given configuration.

    ## Args:
    * `model_cfg` (`ModelConfiguration`): The model configuration.
    """
    criterion: nn.Module = hydra_instantiate(cfg=model_cfg.criterion, **kwargs)
    return criterion.to(device)


def instantiate_optimizer(model_cfg: ModelConfiguration, model: nn.Module, **kwargs):
    """
    Instantiates the optimizer from a given configuration.

    ## Args:
    * `model_cfg` (`ModelConfiguration`): The model configuration.
    * `model` (`nn.Module`): The model to optimize.
    """
    optimizer: torch.optim.Optimizer = hydra_instantiate(
        cfg=model_cfg.optimizer, params=model.parameters(), **kwargs
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
