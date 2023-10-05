# imports
from hydra.utils import instantiate
from omegaconf import DictConfig

# torch imports
import torch
import torch.nn as nn

# monai
from generative.inferers import DiffusionInferer
from generative.networks.schedulers import Scheduler

# research libraries
from coca_pytorch.coca_pytorch import CoCa
from vit_pytorch.extractor import Extractor
from vit_pytorch.simple_vit_with_patch_dropout import SimpleViT

# rtk
from rtk.config import Configuration, ModelConfiguration, DiffusionModelConfiguration
from rtk.utils import get_logger, hydra_instantiate

logger = get_logger(__name__)


def get_vit_extractor(cfg: Configuration):
    image_size = cfg.datasets.dim
    vit = SimpleViT(
        depth=6,
        dim=1024,
        heads=16,
        image_size=image_size,
        mlp_dim=2048,
        num_classes=2,
        patch_size=32,  # https://arxiv.org/abs/2212.00794
    )
    vit = Extractor(vit, return_embeddings_only=True, detach=False)
    return vit


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

    if model_name == "CoCa":
        vit = get_vit_extractor(cfg=cfg)
        kwargs["img_encoder"] = vit
    model: nn.Module = hydra_instantiate(cfg=model_cfg.model, **kwargs)
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
