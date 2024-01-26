##!/usr/bin/env python3
import hydra
import math
import os
import random
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from tqdm.auto import tqdm

# torch
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torchvision import transforms
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader

# torchmetrics
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.metric import Metric

# :huggingface:
from accelerate import Accelerator
from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

# monai
import monai

# rtk
from rtk import datasets, repl, models
from rtk.config import *
from rtk.datasets import _IMAGE_KEYNAME, _LABEL_KEYNAME
from rtk.diffusion import *
from rtk.mlflow import *
from rtk.utils import hydra_instantiate, get_logger, _strip_target

_MAX_RAND_INT = 8192


def train_loop(
    cfg: Configuration,
    train_loader: DataLoader,
    test_loader: DataLoader = None,
    **start_run_kwargs,
):
    # Initialize accelerator and tensorboard logging
    model_cfg = cfg.models
    output_dir = "artifacts"
    max_epochs = cfg.job.max_epochs
    ignite_cfg = cfg.ignite
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
        device_placement=False,
    )
    device = torch.device(cfg.job.device)
    logger.info(f"Using device:\t'{device}'")
    use_multi_gpu = cfg.job.get("use_multi_gpu", False)
    if accelerator.is_main_process:
        accelerator.init_trackers("run_diffusion")

    # Prepare everything
    # prepare model
    model = models.instantiate_model(cfg, device=device)
    optimizer = models.instantiate_optimizer(cfg, model=model)
    noise_scheduler = models.instantiate_diffusion_scheduler(cfg)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_loader) * cfg.job.max_epochs),
    )
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        lr_scheduler,
        device_placement=[False, False, False, False],
    )

    global_step = 0
    log_interval = ignite_cfg.get("log_interval", max(cfg.job.max_epochs // 10, 1))

    # Now you train the model
    for epoch in range(max_epochs):
        progress_bar = tqdm(
            total=len(train_loader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch+1}")

        for batch in train_loader:
            if use_multi_gpu:
                train_loader.sampler.set_epoch(epoch)

            clean_images: torch.Tensor = batch[0].to(device)
            noise, timesteps, noisy_images = forward_diffusion(
                clean_images,
                model_cfg,
                noise_scheduler,
            )

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            mlflow.log_metrics(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            pipeline = DDPMPipeline(unet=unwrapped_model, scheduler=noise_scheduler)

            if (epoch + 1) % log_interval == 0 or epoch == max_epochs - 1:
                evaluate(
                    cfg,
                    epoch,
                    pipeline,
                    loader=test_loader,
                    device=device,
                )

            if (epoch + 1) % log_interval == 0 or epoch == max_epochs - 1:
                state_dict = (
                    unwrapped_model.state_dict()
                    if not use_multi_gpu
                    else unwrapped_model.module.state_dict()
                )
                torch.save(state_dict, f"{output_dir}/{(epoch+1):04d}-model.pt")

    logger.info("Training finished.")


def run_loop(
    cfg: Configuration,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device = torch.device("cuda"),
):
    if cfg.job.use_azureml:
        log_mlflow_params(cfg)

    if cfg.job.mode == "diffusion":
        train_loop(cfg, train_loader, test_loader=test_loader)

    elif cfg.job.mode == "diffusion-evaluate":
        # prepare model and pipeline
        model = models.instantiate_model(cfg, device=device)
        noise_scheduler = models.instantiate_diffusion_scheduler(cfg)
        pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
        evaluate(
            cfg,
            epoch=-1,
            pipeline=pipeline,
            loader=test_loader,
            device=device,
        )

    if cfg.job.use_azureml:
        mlflow.log_artifact("./")
        mlflow.end_run()


@hydra.main(version_base=None, config_path="", config_name="")
def main(cfg: Configuration, **kwargs):
    # before we run....
    logger.debug(OmegaConf.to_yaml(cfg))
    job_cfg = cfg.job
    random_state: int = job_cfg.get("random_state", random.randint(0, _MAX_RAND_INT))
    use_multi_gpu = job_cfg.get("use_multi_gpu", False)

    monai.utils.set_determinism(seed=random_state)
    logger.info(f"Using seed:\t{random_state}")

    # device = torch.device(job_cfg.device)

    run_name = create_run_name(cfg=cfg, random_state=random_state)
    logger.info(f"Run name:\t'{run_name}'")

    if use_multi_gpu:
        rank: int = kwargs["rank"]
        world_size: int = kwargs["world_size"]
        models.ddp_setup(rank, world_size)

    # prepare data
    loaders = datasets.prepare_data(cfg)
    train_loader = loaders[0]
    test_loader = loaders[-1]

    os.makedirs("artifacts", exist_ok=True)

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
            run_loop(cfg, train_loader, test_loader=test_loader)

    else:
        run_loop(cfg, train_loader, test_loader=test_loader)

    if use_multi_gpu:
        destroy_process_group()


if __name__ == "__main__":
    repl.install(show_locals=False)
    logger = get_logger("rtk.scripts")
    monai.config.print_config()
    if False:
        world_size: int = torch.cuda.device_count()
        mp.spawn(main, args=(world_size), nprocs=world_size)
    else:
        main()
