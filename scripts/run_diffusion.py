##!/usr/bin/env python3
import logging
import os
import random
import sys
import hydra
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
from tqdm.auto import tqdm

# torch
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader

# torchmetrics
from torchmetrics.metric import Metric

# :huggingface:
from accelerate import Accelerator
from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import HfFolder, Repository, whoami

# monai
import monai

# rtk
from rtk import datasets, repl, models
from rtk.datasets import _IMAGE_KEYNAME, _LABEL_KEYNAME
from rtk.config import *
from rtk.mlflow import *
from rtk.utils import hydra_instantiate, get_logger, _strip_target

_MAX_RAND_INT = 8192


def instantiate_torch_metrics(cfg: Configuration, **kwargs):
    ignite_cfg = cfg.ignite
    metrics = []
    for metric_cfg in ignite_cfg.metrics:
        metric = hydra_instantiate(metric_cfg, **kwargs)
        target_name = _strip_target(metric_cfg, lower=False)
        if target_name == "FrechetInceptionDistance":
            new_target_name = "FID"

        elif target_name == "InceptionScore":
            new_target_name = "IS"

        elif target_name == "MultiScaleStructuralSimilarityIndexMeasure":
            new_target_name = "MS-SSIM"

        else:
            raise ValueError(f"Unknown metric: '{target_name}'")

        metrics.append({new_target_name: metric})

    return metrics


# https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb
def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in tqdm(enumerate(images), desc="Creating grid"):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def forward_diffusion(
    clean_images: torch.Tensor,
    model_cfg: Configuration,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
):
    # Sample noise to add to the images
    noise = torch.randn(clean_images.shape).to(clean_images.device)
    bs = clean_images.shape[0]

    # Sample a random timestep for each image
    timesteps = torch.randint(
        0,
        model_cfg.scheduler.num_train_timesteps,
        (bs,),
        device=clean_images.device,
    ).long()

    # Add noise to the clean images according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
    return noise, timesteps, noisy_images


# https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb
def evaluate(
    cfg: Configuration,
    epoch: int,
    pipeline: DDPMPipeline,
    loader: DataLoader = None,
    log_samples: bool = True,
):
    logger.info("Evaluating model...")

    dataset_cfg = cfg.datasets
    # metrics = instantiate_torch_metrics(cfg)

    # pred_images = pipeline(
    #     batch_size=len(loader.dataset),
    #     generator=loader,
    # ).images

    # def _update_metric(key: str, metric: Metric, images: DataLoader, **kwargs):
    #     for image, _ in loader:
    #         if key == "MS-SSIM":
    #             logger.warn("'MS-SSIM' is not yet implemented. Skipping calculation...")

    #         elif key == "FID":
    #             logger.warn("'FID' is not yet implemented. Skipping calculation...")
    #             # metric.update(image, real=True)
    #             # metric.update(image, real=False)

    #         elif key == "IS":
    #             metric.update(image, **kwargs)

    #     out = metric.compute()
    #     return out

    # logged_metrics = {}
    # for item in metrics:
    #     key = list(item.keys())[0]
    #     metric = item[key]
    #     out: torch.Tensor = _update_metric(key, metric, loader)
    #     if key == "IS":
    #         key_mean = f"{key}_mean"
    #         key_std = f"{key}_std"
    #         logged_metrics[key_mean] = out[0].item()
    #         logged_metrics[key_std] = out[1].item()
    #     else:
    #         logged_metrics[key] = out.item()

    # mlflow.log_metrics(logged_metrics, step=epoch)

    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    if log_samples:
        samples_images = pipeline(
            batch_size=dataset_cfg.dataloader.batch_size,
            generator=torch.manual_seed(cfg.job.random_state),
        ).images

        # Make a grid out of the images
        image_grid = make_grid(samples_images, rows=4, cols=4)

        # Save the images
        test_dir = os.path.join("artifacts", "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{(epoch+1):04d}.png")


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
        accelerator.init_trackers("train_example")

    # Prepare everything
    # prepare model
    model = models.instantiate_model(cfg, device=device)
    optimizer = models.instantiate_optimizer(cfg, model=model)
    # criterion = models.instantiate_criterion(cfg, device=device)
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
    # model.to(device)

    global_step = 0
    log_interval = ignite_cfg.get("log_interval", max(cfg.job.max_epochs // 10, 1))

    # Now you train the model
    for epoch in range(max_epochs):
        progress_bar = tqdm(
            total=len(train_loader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch+1}")

        for _, batch in enumerate(train_loader):
            if use_multi_gpu:
                train_loader.sampler.set_epoch(epoch)

            clean_images: torch.Tensor = batch[0].to(device)
            noise, timesteps, noisy_images = forward_diffusion(
                clean_images,
                model_cfg,
                device,
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
            # accelerator.log(logs, step=global_step)
            mlflow.log_metrics(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            pipeline = DDPMPipeline(unet=unwrapped_model, scheduler=noise_scheduler)

            if (epoch + 1) % log_interval == 0 or epoch == max_epochs - 1:
                evaluate(cfg, epoch, pipeline, loader=test_loader)

            if (epoch + 1) % log_interval == 0 or epoch == max_epochs - 1:
                state_dict = (
                    unwrapped_model.state_dict()
                    if not use_multi_gpu
                    else unwrapped_model.module.state_dict()
                )
                torch.save(
                    unwrapped_model.state_dict(), f"{output_dir}/{epoch:04d}-model.pt"
                )

    logger.info("Training finished.")


@hydra.main(version_base=None, config_path="", config_name="")
def main(cfg: Configuration, **kwargs):
    # before we run....
    logger.debug(OmegaConf.to_yaml(cfg))
    job_cfg = cfg.job
    mode: str = job_cfg.get("mode", "diffusion")
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
            log_mlflow_params(cfg)
            train_loop(cfg, train_loader, test_loader=test_loader)
            mlflow.log_artifact("./")
    else:
        train_loop(cfg, train_loader, test_loader=test_loader)

    mlflow.end_run()

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
