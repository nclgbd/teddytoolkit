##!/usr/bin/env python
"""Script to train a model using Hydra configuration."""
import logging
import os
import random
import sys
from omegaconf import OmegaConf

# hydra
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd

# azureml
from azureml.core import Run, Experiment

# mlflow
import mlflow

# torch
import torch
from torch.utils.data import DataLoader

# monai
import monai

# rtk
from rtk import datasets, repl
from rtk.config import (
    Configuration,
    DatasetConfiguration,
    JobConfiguration,
    ModelConfiguration,
)
from rtk.ignite import prepare_run, evaluate
from rtk.mlflow import *
from rtk.utils import hydra_instantiate, get_logger

_MAX_RAND_INT = 8192


def run_trainer(
    loaders: list, train_loader: DataLoader, device: torch.device, cfg: Configuration
):
    job_cfg: JobConfiguration = cfg.job
    prepare_func: callable = hydra_instantiate(
        job_cfg["prepare_function"], _partial_=True
    )
    trainer, _ = prepare_func(loaders=loaders, device=device, cfg=cfg)
    cfg.datasets.labels = datasets.set_labels_from_encoding(cfg)
    state = trainer.run(
        data=train_loader,
        max_epochs=job_cfg.max_epochs,
        epoch_length=job_cfg.epoch_length,
    )
    return state


def run_evaluate(loaders: list, device: torch.device, cfg: Configuration):
    prepare_run(loaders=loaders, device=device, cfg=cfg, mode="evaluate")


def run_train(
    cfg: Configuration,
    run_name: str,
    loaders: list,
    train_loader: DataLoader,
    device: torch.device,
):
    if cfg.job.use_mlflow:
        start_run_kwargs = prepare_mlflow(cfg)
        with mlflow.start_run(
            run_name=run_name,
            **start_run_kwargs,
        ) as mlflow_run:
            logger.debug(
                "run_id: {}, status: {}".format(
                    mlflow_run.info.run_id, mlflow_run.info.status
                )
            )
            log_mlflow_params(cfg)
            state = run_trainer(loaders, train_loader, device, cfg)
            mlflow.log_artifact("./")
    else:
        state = run_trainer(loaders, train_loader, device, cfg)

    return state


def run_eval(
    cfg: Configuration,
    run_name: str,
    loaders: list,
    device: torch.device,
):
    if cfg.job.use_mlflow:
        start_run_kwargs = prepare_mlflow(cfg)
        with mlflow.start_run(
            run_name=run_name,
            **start_run_kwargs,
        ) as mlflow_run:
            logger.debug(
                "run_id: '{}'; status: '{}'".format(
                    mlflow_run.info.run_id, mlflow_run.info.status
                )
            )
            log_mlflow_params(cfg)
            run_evaluate(loaders, device, cfg)
            mlflow.log_artifact("./")
    else:
        run_evaluate(loaders, device, cfg)
    return


@hydra.main(version_base=None, config_path="", config_name="")
def main(cfg: Configuration) -> None:
    dataset_cfg: DatasetConfiguration = cfg.datasets
    # before we run....
    logger.debug(OmegaConf.to_yaml(cfg))
    mode: str = cfg.get("mode", "train")
    random_state: int = cfg.get("random_state", random.randint(0, _MAX_RAND_INT))

    monai.utils.set_determinism(seed=random_state)
    logger.info(f"Using seed:\t{random_state}")

    device = torch.device(cfg.device)
    logger.info(f"Using device:\t{device}")

    run_name = create_run_name(cfg=cfg, random_state=random_state)
    logger.info(f"Run name:\t'{run_name}'")

    # prepare data
    loaders = datasets.prepare_validation_dataloaders(cfg)
    train_loader = loaders[0]
    test_loader = loaders[-1]

    os.makedirs("artifacts", exist_ok=True)

    # run trainer
    if mode == "train":
        run_train(cfg, run_name, loaders, train_loader, device)

    elif mode == "evaluate":
        run_eval(cfg, run_name, loaders, device)

    else:
        raise ValueError(f"Unknown mode: '{mode}'")

    logger.info("Job complete.")


if __name__ == "__main__":
    repl.install(show_locals=False)
    logger = get_logger("rtk.scripts")
    monai.config.print_config()
    main()
