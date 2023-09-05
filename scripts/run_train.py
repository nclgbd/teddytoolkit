##!/usr/bin/env python
"""Script to train a model using Hydra configuration."""
import hydra
import logging
import os
import sys
from omegaconf import OmegaConf

# torch
import torch
from torch.utils.data import DataLoader

# ignite
from ignite.engine import Engine, create_supervised_trainer

# ignite.contrib
from ignite.contrib.handlers import ProgressBar

# monai
import monai

# sys path append
CWD = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(CWD)))

# ttk
from ttk import datasets, models, repl
from ttk.config import (
    Configuration,
    DatasetConfiguration,
    JobConfiguration,
)
from ttk.ignite import create_default_trainer_args, prepare_run
from ttk.utils import hydra_instantiate, get_logger


def create_loaders(cfg: Configuration):
    """Create train and test loaders."""
    dataset_cfg: DatasetConfiguration = cfg.datasets
    job_cfg: JobConfiguration = cfg.job
    train_transform = datasets.create_transforms(
        dataset_cfg, use_transforms=job_cfg.use_transforms
    )
    eval_transform = datasets.create_transforms(dataset_cfg, use_transforms=False)
    dataset = datasets.instantiate_image_dataset(cfg=cfg, transform=train_transform)
    train_dataset, test_dataset = dataset[0], dataset[1]
    train_val_test_split_dict = datasets.instantiate_train_val_test_datasets(
        cfg=cfg, dataset=train_dataset
    )
    train_dataset, val_dataset = (
        train_val_test_split_dict["train"],
        train_val_test_split_dict["val"],
    )
    train_dataset.transform = train_transform
    test_dataset.transform = eval_transform
    train_loader: DataLoader = hydra_instantiate(
        cfg=dataset_cfg.dataloader,
        dataset=train_dataset,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
    )
    val_loader: DataLoader = hydra_instantiate(
        cfg=dataset_cfg.dataloader,
        dataset=val_dataset,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
    )
    test_loader: DataLoader = hydra_instantiate(
        cfg=dataset_cfg.dataloader,
        dataset=test_dataset,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
    )
    return train_loader, val_loader, test_loader


@hydra.main(version_base=None, config_path="", config_name="")
def main(cfg: Configuration) -> None:
    # before we run....
    logger.debug(OmegaConf.to_yaml(cfg))
    job_cfg: JobConfiguration = cfg.job
    monai.utils.set_determinism(seed=job_cfg.random_state)
    logger.info(f"Using seed: {job_cfg.random_state}")

    device = torch.device(job_cfg.device)
    logger.info(f"Using device: {job_cfg.device}")

    # prepare data
    loaders = create_loaders(cfg)

    # prepare run
    trainer, evaluators = prepare_run(cfg=cfg, loaders=loaders, device=device)

    # # run trainer
    train_loader = loaders[0]
    state = trainer.run(
        data=train_loader,
        max_epochs=job_cfg.max_epochs,
        epoch_length=job_cfg.epoch_length,
    )
    return state


if __name__ == "__main__":
    repl.install(show_locals=False)
    logger = get_logger("run_train")
    monai.config.print_config()
    main()
