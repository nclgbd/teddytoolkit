##!/usr/bin/env python
"""Script to train a model using Hydra configuration."""
import logging
import os
import sys
import random
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

# sys path append
CWD = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(CWD)))

# rtk
from rtk import datasets, repl
from rtk.config import (
    Configuration,
    DatasetConfiguration,
    JobConfiguration,
    ModelConfiguration,
)
from rtk.ignite import prepare_run
from rtk.utils import hydra_instantiate, get_logger, login, create_run_name


def create_loaders(cfg: Configuration):
    """Create train and test loaders."""
    dataset_cfg: DatasetConfiguration = cfg.datasets
    job_cfg: JobConfiguration = cfg.job
    train_transform = datasets.create_transforms(
        cfg, use_transforms=job_cfg.use_transforms
    )
    eval_transform = datasets.create_transforms(cfg, use_transforms=False)
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


def prepare_mlflow(cfg: Configuration):
    logger.info("Starting MLflow run...")
    mlflow_cfg = cfg.mlflow
    if cfg.job.use_azureml:
        logger.info("Using AzureML for experiment tracking...")
        ws = login()
        tracking_uri = ws.get_mlflow_tracking_uri()

    else:
        tracking_uri = mlflow_cfg.get("tracking_uri", "~/mlruns/")

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = HydraConfig.get().job.config_name
    experiment_id = mlflow.create_experiment(
        experiment_name, artifact_location=tracking_uri
    )
    logger.debug(f"MLflow tracking URI: {tracking_uri}")
    start_run_kwargs: dict = mlflow_cfg.get("start_run", {})
    start_run_kwargs["experiment_id"] = experiment_id
    return start_run_kwargs


@hydra.main(version_base=None, config_path="", config_name="")
def main(cfg: Configuration) -> None:
    # before we run....
    logger.debug(OmegaConf.to_yaml(cfg))
    job_cfg: JobConfiguration = cfg.job
    random_state: int = job_cfg.get("random_state", random.randint(0, 8192))
    run_name = create_run_name(cfg=cfg, random_state=random_state)
    logger.info(f"Run name: {run_name}")
    monai.utils.set_determinism(seed=random_state)
    logger.info(f"Using seed: {random_state}")

    device = torch.device(job_cfg.device)
    logger.info(f"Using device: {job_cfg.device}")

    # prepare data
    loaders = create_loaders(cfg)
    train_loader = loaders[0]

    def run_trainer():
        # prepare_function_kwargs: dict = job_cfg.get("prepare_function", {})
        # prepare_function_kwargs["cfg"] = cfg
        # prepare_function_kwargs["loaders"] = loaders
        # prepare_function_kwargs["device"] = device
        # trainer, _ = prepare_run(**prepare_function_kwargs)
        trainer, _ = prepare_run(loaders=loaders, device=device, cfg=cfg)
        state = trainer.run(
            data=train_loader,
            max_epochs=job_cfg.max_epochs,
            epoch_length=job_cfg.epoch_length,
        )
        return state

    # run trainer
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
            state = run_trainer()
            mlflow.log_artifact("./")
    else:
        state = run_trainer()

    return state


if __name__ == "__main__":
    repl.install(show_locals=False)
    logger = get_logger("rtk.scripts.run_train")
    monai.config.print_config()
    main()
