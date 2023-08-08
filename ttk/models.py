# imports
from hydra.utils import instantiate
from omegaconf import DictConfig

# torch imports
import torch
import torch.nn as nn

# monai imports
import monai.networks.nets as mnn

# teddytoolkit
from ttk.config import Configuration, ModelConfiguration, DiffusionModelConfiguration
from ttk.utils import get_logger, hydra_instantiate

logger = get_logger(__name__)


def instantiate_model(
    model_cfg: ModelConfiguration, device: torch.device = torch.device("cpu"), **kwargs
):
    return hydra_instantiate(model_cfg, **kwargs).to(device)


def instantiate_diffusion_model(
    diffusion_model_cfg: DiffusionModelConfiguration,
    device: torch.device = torch.device("cpu"),
    **kwargs
):
    return hydra_instantiate(diffusion_model_cfg, **kwargs).to(device)
