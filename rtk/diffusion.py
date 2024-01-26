"""Diffusion related opertations."""

import math
import numpy as np
import os
import random
from PIL import Image
from copy import deepcopy
from tqdm.auto import tqdm

# torch
import torch
from torchvision import transforms

# torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure

# :huggingface:
from accelerate import Accelerator
from diffusers import DDPMPipeline, DDPMScheduler

# monai
import monai

# mlflow
import mlflow

# rtk
from rtk import datasets
from rtk.config import Configuration
from rtk.utils import hydra_instantiate, _strip_target, get_logger

logger = get_logger(__name__)


def instantiate_torch_metrics(cfg: Configuration, **kwargs):
    ignite_cfg = cfg.ignite
    metrics = {}
    for metric_cfg in ignite_cfg.metrics:
        metric = hydra_instantiate(metric_cfg, **kwargs)
        target_name = _strip_target(metric_cfg, lower=False)
        if target_name == "FrechetInceptionDistance":
            new_target_name = "fid"

        elif target_name == "InceptionScore":
            new_target_name = "inception"

        elif target_name == "MultiScaleStructuralSimilarityIndexMeasure":
            new_target_name = "ms-ssim"

        else:
            raise ValueError(f"Unknown metric: '{target_name}'")

        metrics[new_target_name] = metric

    return metrics


# https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb
def make_grid(images):
    dim = int(math.sqrt(len(images)))
    rows, cols = dim, dim
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in tqdm(enumerate(images), desc="Creating grid"):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


# https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb
def evaluate(
    cfg: Configuration,
    epoch: int,
    pipeline: DDPMPipeline,
    loader: monai.data.DataLoader,
    device: torch.device,
    num_samples: int = 32,
):
    logger.info("Evaluating model...")

    old_dim = cfg.datasets.dim
    cfg.datasets.dim = 299
    eval_transforms = datasets.create_transforms(cfg, use_transforms=False)
    dataset = deepcopy(loader.dataset)
    dataset.transform = eval_transforms
    metrics = instantiate_torch_metrics(cfg)
    fid_metric: FrechetInceptionDistance = metrics["fid"]
    inception_metric: InceptionScore = metrics["inception"]
    ms_ssim_metric: MultiScaleStructuralSimilarityIndexMeasure = metrics["ms-ssim"]
    # ms_ssims = []

    # get `num_samples` real images from dataset
    logger.info("Getting real images...")
    real_images = []
    for batch in loader:
        b_images = list(batch[0].to("cpu"))
        real_images.extend(b_images)

        if len(real_images) >= num_samples:
            break

    # get `num_samples` fake images from dataset
    generator = torch.Generator(device=device).manual_seed(cfg.job.random_state)
    fake_images = generate_samples(
        cfg,
        pipeline,
        device=device,
        epoch=epoch,
        generator=generator,
        num_samples=num_samples,
        save_images=True,
    )
    # fake_eval_transforms = transforms.Compose(eval_transforms.transforms[:-2])
    fake_images = [eval_transforms(np.array(img)) for img in fake_images]
    fake_images = torch.Tensor(fake_images).permute(0, 3, 1, 2)
    # fake_images = torch.Tensor(fake_images)

    # Compute metrics
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    fid = fid_metric.compute()

    inception_metric.update(fake_images)
    inception_m, inception_s = inception_metric.compute()
    ms_ssim = ms_ssim_metric(fake_images, real_images)

    # Log metrics
    logged_metrics = {
        "fid": fid,
        "inception_mean": inception_m,
        "inception_std": inception_s,
        "ms_ssim": ms_ssim,
    }
    cfg.datasets.dim = old_dim

    if cfg.job.use_azureml:
        mlflow.log_metrics(logged_metrics, step=epoch)


def generate_samples(
    cfg: Configuration,
    pipeline: DDPMPipeline,
    device: torch.device,
    epoch: int = 0,
    generator: torch.Generator = None,
    num_samples: int = 32,
    save_images: bool = False,
    **kwargs,
):
    logger.info("Generating samples...")
    generator = (
        torch.Generator(device=device).manual_seed(cfg.job.random_state)
        if generator is None
        else generator
    )

    batch_size = cfg.datasets.dataloader.batch_size
    samples_images = []

    # The default pipeline output type is `List[PIL.Image]`
    for _ in range(num_samples):
        pipe_images = pipeline(
            batch_size=batch_size,
            generator=generator,
            **kwargs,
        ).images

        samples_images.extend(pipe_images)

        if len(samples_images) >= num_samples:
            break

        generator = torch.Generator(device=device).manual_seed(random.randint(0, 8192))

    if save_images:
        # Make a grid out of the images
        random_samples = random.choices(samples_images, k=batch_size)
        image_grid = make_grid(random_samples)

        # Save the images
        test_dir = os.path.join("artifacts", "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{(epoch+1):04d}.png")

    return samples_images


def forward_diffusion(
    clean_images: torch.Tensor,
    model_cfg: Configuration,
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
