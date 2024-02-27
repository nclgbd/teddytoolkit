"""Diffusion related opertations."""

import math
import numpy as np
import os
import random
from PIL import Image
from copy import deepcopy
from tqdm.auto import tqdm
from packaging import version

# torch
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

# :huggingface:
import accelerate
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from diffusers.training_utils import EMAModel, cast_training_params
from diffusers import (
    UNet2DConditionModel,
    DiffusionPipeline,
    DDPMScheduler,
    AutoencoderKL,
)
from diffusers.utils.import_utils import is_xformers_available

# transformers
from transformers import CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.utils import ContextManagers

# peft
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

# mlflow
import mlflow

# rtk
from rtk import datasets
from rtk.config import *
from rtk.utils import hydra_instantiate, _strip_target, get_logger

logger = get_logger(__name__)


def instantiate_torch_metrics(tm_cfg: TorchMetricsConfiguration, remap=True, **kwargs):
    _metrics = tm_cfg.metrics
    metrics = {}
    for metric_cfg in _metrics:
        metric = hydra_instantiate(metric_cfg, **kwargs)
        target_name = _strip_target(metric_cfg, lower=False)
        if remap:
            try:
                target_name: str = tm_cfg.remap[target_name]

            except Exception as e:
                if isinstance(e, KeyError):
                    logger.warning(
                        f"No remap found for '{target_name}', using default name."
                    )
                if isinstance(e, AttributeError):
                    logger.warning(
                        f"No 'remap' attribute found in torch metrics configuration, using default name."
                    )
                    logger.debug(tm_cfg)
        else:
            raise ValueError(f"Unknown metric: '{target_name}'")

        target_name = target_name.lower()
        metrics[target_name] = metric

    return metrics


def compile_huggingface_pipeline(
    cfg: TextToImageConfiguration,
    accelerator: Accelerator,
    device: torch.device = None,
    weight_dtype: torch.dtype = torch.float16,
):
    logger.info("Compiling HuggingFace pipeline...")
    hf_cfg = cfg.huggingface

    device = device if device is not None else accelerator.device

    unet: UNet2DConditionModel = hydra_instantiate(hf_cfg.unet)
    scheduler: DDPMScheduler = hydra_instantiate(
        hf_cfg.scheduler, torch_dtype=weight_dtype
    )
    tokenizer: CLIPTokenizer = hydra_instantiate(
        hf_cfg.tokenizer, torch_dtype=weight_dtype
    )
    pipeline: DiffusionPipeline = hydra_instantiate(
        hf_cfg.pipeline, torch_dtype=weight_dtype
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = (
            AcceleratorState().deepspeed_plugin
            if accelerate.state.is_initialized()
            else None
        )
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
        # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
        # will try to assign the same optimizer with the same weights to all models during
        # `deepspeed.initialize`, which of course doesn't work.
        #
        # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
        # frozen models from being partitioned during `zero.Init` which gets called during
        # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
        # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
        text_encoder: CLIPTextModel = hydra_instantiate(
            hf_cfg.text_encoder, torch_dtype=weight_dtype
        )
        vae: AutoencoderKL = hydra_instantiate(hf_cfg.vae, torch_dtype=weight_dtype)

    # Freeze vae and text_encoder and set unet to trainable
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Prepare the LoRA adapter for the UNet
    # Freeze the unet parameters before adding adapters
    for param in unet.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)
    if cfg.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    # Create EMA for the unet.
    ema_unet: torch.nn.Module = None
    if cfg.use_ema:
        logger.warn("Using EMA for the UNet")
        ema_unet: torch.nn.Module = hydra_instantiate(hf_cfg.unet)
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=ema_unet.config,
        )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if cfg.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if cfg.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DConditionModel
                )
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    return pipeline, unet, scheduler, tokenizer, text_encoder, vae, ema_unet


# https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb
def make_grid(images, dim: int = None):
    dim = int(math.sqrt(len(images))) if dim is None else dim
    rows, cols = dim, dim
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in tqdm(enumerate(images), desc="Creating grid"):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


# https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb
def evaluate(
    cfg: DiffusionConfiguration,
    epoch: int,
    pipeline: DiffusionPipeline,
    loader: DataLoader,
    device: torch.device,
    num_samples: int = None,
    **kwargs,
):
    logger.info("Evaluating model...")
    if True:
        logger.warn("Skipping metric evaluation")

        num_samples = (
            num_samples
            if num_samples is not None
            else cfg.datasets.dataloader.batch_size
        )
        generator = torch.Generator(device=device).manual_seed(epoch)
        fake_images = generate_samples(
            cfg,
            pipeline,
            device=device,
            epoch=epoch,
            generator=generator,
            num_samples=num_samples,
            save_images=True,
            **kwargs,
        )

    else:
        job_cfg = cfg.job
        old_dim = cfg.datasets.dim
        cfg.datasets.dim = 299
        eval_transforms = datasets.create_transforms(cfg, use_transforms=False)
        eval_loader: DataLoader = deepcopy(loader)
        eval_loader.dataset.transform = eval_transforms
        metrics = instantiate_torch_metrics(cfg.torchmetrics)
        fid_metric: FrechetInceptionDistance = metrics["fid"]
        inception_metric: InceptionScore = metrics["inception"]
        ssim_metric: StructuralSimilarityIndexMeasure = metrics["ssim"]
        # ms_ssims = []

        # get `num_samples` real images from dataset
        logger.info("Getting real images...")
        real_images = []
        for batch in eval_loader:
            b_images = list(batch[0].to("cpu"))
            real_images.extend(b_images)

            if len(real_images) >= num_samples:
                break

        # get `num_samples` fake images from dataset
        generator = torch.Generator(device=device).manual_seed(cfg.random_state)
        fake_images = generate_samples(
            cfg,
            pipeline,
            device=device,
            epoch=epoch,
            generator=generator,
            num_samples=num_samples,
            save_images=True,
        )
        fake_img_transforms = [transforms.PILToTensor()]
        fake_img_transforms.extend(eval_transforms.transforms)
        eval_transforms.transforms = fake_img_transforms
        fake_images = [eval_transforms(img) for img in fake_images]
        # fake_images = torch.Tensor(fake_images)

        # Compute metrics
        real_images = torch.stack(real_images)
        fake_images = torch.stack(fake_images)

        fid_metric.update(real_images, real=True)
        fid_metric.update(fake_images, real=False)
        fid = fid_metric.compute()

        inception_metric.update(fake_images)
        inception_m, inception_s = inception_metric.compute()
        ssim = ssim_metric(fake_images, real_images)

        # Log metrics
        logged_metrics = {
            "fid": fid,
            "inception_mean": inception_m,
            "inception_std": inception_s,
            "ssim": ssim,
        }
        cfg.datasets.dim = old_dim

        if job_cfg.use_azureml:
            mlflow.log_metrics(logged_metrics, step=epoch)


def generate_samples(
    cfg: Configuration,
    pipeline: DiffusionPipeline,
    device: torch.device,
    epoch: int = 0,
    generator: torch.Generator = None,
    num_samples: int = 32,
    save_images: bool = False,
    **kwargs,
):
    logger.info("Generating samples...")
    generator = (
        torch.Generator(device=device).manual_seed(cfg.random_state)
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
        img_path = f"{test_dir}/{(epoch+1):04d}.png"
        image_grid.save(img_path)
        mlflow.log_artifact(img_path)

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
        noise_scheduler.config.num_train_timesteps,
        (bs,),
        device=clean_images.device,
    ).long()

    # Add noise to the clean images according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
    return noise, timesteps, noisy_images
