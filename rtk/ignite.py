# imports
from typing import Callable, Union
import hydra
import logging
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import pandas as pd
from rich import inspect

# experiment managing
## azureml
from azureml.core import Experiment, Workspace

## mlflow
import mlflow

# sklearn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as torch_lr_schedulers
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

## ignite.engine
from ignite.engine import (
    Engine,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite import metrics as ignite_metrics_module
from ignite.engine.events import Events
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.handlers.early_stopping import EarlyStopping
from ignite.handlers.param_scheduler import LRScheduler
from ignite.metrics import Metric
from ignite.utils import setup_logger

## ignite.contrib
from ignite.contrib import metrics as c_ignite_metrics_module
from ignite.contrib.handlers import ProgressBar

IGNITE_METRICS_MODULE = [ignite_metrics_module, c_ignite_metrics_module]

# monai
import monai
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    MeanAbsoluteError,
    MeanSquaredError,
    StatsHandler,
    ValidationHandler,
    from_engine,
)

# generative
from generative.engines import DiffusionPrepareBatch
from generative.inferers import DiffusionInferer
from generative.networks.schedulers import Scheduler

# rtk
from rtk import models
from rtk.config import (
    Configuration,
    DatasetConfiguration,
    DiffusionModelConfiguration,
    IgniteConfiguration,
    JobConfiguration,
    ModelConfiguration,
)
from rtk.utils import get_logger, login, hydra_instantiate

logger = get_logger(__name__)


def create_default_trainer_args(
    cfg: Configuration,
    **kwargs,
):
    """
    Prepares the data for the ignite trainer.
    """
    logger.info("Creating default trainer arguments...")
    trainer_kwargs = dict()
    job_cfg: JobConfiguration = kwargs.get("job_cfg", cfg.job)
    model_cfg: ModelConfiguration = kwargs.get("model_cfg", cfg.models)
    device: torch.device = kwargs.get("device", torch.device(job_cfg.device))
    trainer_kwargs["device"] = device

    # Prepare model, optimizer, loss function, and criterion
    model: nn.Module = models.instantiate_model(cfg, device=device)
    model.train()
    trainer_kwargs["model"] = model
    criterion = models.instantiate_criterion(cfg, device=device)
    trainer_kwargs["loss_fn"] = criterion
    optimizer: torch.optim.Optimizer = models.instantiate_optimizer(cfg, model=model)
    trainer_kwargs["optimizer"] = optimizer

    return trainer_kwargs


def create_diffusion_model_evaluator(
    cfg: Configuration, model: nn.Module, inferer: DiffusionInferer, **kwargs
):
    """
    Creates the validation/test evaluator for the diffusion model.

    ## Args:
    """
    device: torch.device = kwargs.get("device", torch.device(cfg.job.device))
    use_autocast: bool = kwargs.get("use_autocast", cfg.job.use_autocast)

    def diffusion_validation_step(
        engine: Engine,
        batch: dict,
    ):
        """
        The validation step function for the diffusion model.

        Sources:
        - https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/3d_ddpm/3d_ddpm_tutorial.ipynb.
        - https://pytorch-ignite.ai/how-to-guides/02-convert-pytorch-to-ignite/

        """
        model.eval()
        # with torch.no_grad():
        #     x, y = batch
        #     y_pred = model(x)

        # return y_pred, y
        images, _ = batch
        images: torch.Tensor = images.to(device)
        noise = torch.randn_like(images).to(device)
        with torch.no_grad():
            with autocast(enabled=use_autocast):
                timesteps = torch.randint(
                    0,
                    inferer.scheduler.num_train_timesteps,
                    (images.shape[0],),
                    device=device,
                ).long()

                # Get model prediction
                noise_pred = inferer(
                    inputs=images,
                    diffusion_model=model,
                    noise=noise,
                    timesteps=timesteps,
                )
                val_loss = F.mse_loss(noise_pred.float(), noise.float())

        return val_loss

    evaluator = Engine(diffusion_validation_step)
    return evaluator


def create_diffusion_model_trainer(
    cfg: Configuration,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inferer: DiffusionInferer,
    **kwargs,
):
    kwargs = {} if kwargs is None else kwargs
    # ignite_cfg: IgniteConfiguration = kwargs.get("ignite_cfg", cfg.ignite)
    job_cfg: JobConfiguration = kwargs.get("job_cfg", cfg.job)
    device: torch.device = kwargs.get("device", torch.device(cfg.job.device))
    use_autocast: bool = kwargs.get("use_autocast", job_cfg.use_autocast)

    ##### Prepare trainer #####

    # inputs, targets = batch
    # optimizer.zero_grad()
    # outputs = model(inputs)
    # loss = criterion(outputs, targets)
    # loss.backward()
    # optimizer.step()
    # return loss.item()

    def diffusion_train_step(
        engine: Engine,
        batch: dict,
    ):
        """
        The training step function for the diffusion model.

        Sources:
        - https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/3d_ddpm/3d_ddpm_tutorial.ipynb.
        - https://pytorch-ignite.ai/how-to-guides/02-convert-pytorch-to-ignite/

        """
        if not use_autocast:
            raise NotImplementedError("`auto_cast` required.")
        model.train()
        images, _ = batch
        images: torch.Tensor = images.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_autocast):
            # Generate random noise
            noise = torch.randn_like(images).to(device)

            # Create timesteps
            timesteps: torch.long = torch.randint(
                0,
                inferer.scheduler.num_train_timesteps,
                (images.shape[0],),
                device=images.device,
            ).long()

            # Get model prediction
            noise_pred = inferer(
                inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler = GradScaler()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss

    trainer = Engine(diffusion_train_step)
    return trainer


# def create_diffusion_model_engines(
#     cfg: Configuration,
#     train_loader: torch.utils.data.DataLoader,
#     val_loader: torch.utils.data.DataLoader = None,
#     **kwargs,
# ):
#     ignite_cfg: IgniteConfiguration = kwargs.get("ignite_cfg", cfg.ignite)
#     trainer_kwargs = create_default_trainer_args(cfg=cfg, **kwargs)
#     model, optimizer = trainer_kwargs["model"], trainer_kwargs["optimizer"]
#     engine_dict = dict()
#     job_cfg: JobConfiguration = kwargs.get("job_cfg", cfg.job)
#     model_cfg: ModelConfiguration = kwargs.get("model_cfg", cfg.models)
#     device: torch.device = kwargs.get("device", torch.device(job_cfg.device))
#     scheduler: Scheduler = models.instantiate_diffusion_scheduler(model_cfg)
#     inferer: DiffusionInferer = models.instantiate_diffusion_inferer(
#         model_cfg, scheduler=scheduler
#     )

#     trainer = create_diffusion_model_trainer(
#         cfg=cfg, model=model, optimizer=optimizer, inferer=inferer
#     )
#     engine_dict["trainer"] = trainer
#     train_evaluator = create_diffusion_model_evaluator(
#         cfg=cfg, model=model, inferer=inferer
#     )
#     engine_dict["train_evaluator"] = train_evaluator

#     # add additional handlers
#     ProgressBar().attach(trainer)
#     log_interval = ignite_cfg.get("log_interval", job_cfg.epochs // 10)
#     trainer.add_event_handler(
#         event_name=Events.EPOCH_COMPLETED(every=log_interval),
#         handler=prepare_diffusion_run,
#         **{
#             "evaluator": train_evaluator,
#             "loader": train_loader,
#             "cfg": cfg,
#             # sampling from diffusion model kwargs
#             "model": model,
#             "scheduler": scheduler,
#             "inferer": inferer,
#             "device": device,
#         },
#     )
#     ProgressBar().attach(train_evaluator)

#     if val_loader is not None:
#         val_evaluator = create_diffusion_model_evaluator(
#             cfg=cfg, model=model, inferer=inferer
#         )
#         trainer.add_event_handler(
#             event_name=Events.EPOCH_COMPLETED(every=log_interval),
#             handler=prepare_diffusion_run,
#             **{
#                 "evaluator": val_evaluator,
#                 "loader": val_loader,
#                 "cfg": cfg,
#                 # sample from diffusion model kwargs
#                 "model": model,
#                 "scheduler": scheduler,
#                 "device": device,
#             },
#         )
#         ProgressBar().attach(val_evaluator)
#         engine_dict["val_evaluator"] = val_evaluator

#     logger.info("Diffusion model engines created.\n")
#     return engine_dict


def sample_from_diffusion_model(
    # engine: Engine,
    cfg: Configuration,
    model: nn.Module,
    scheduler: Scheduler,
    inferer: DiffusionInferer,
    device: torch.device,
    num_inference_steps: int = 1000,
    val_score: float = 0.0,
    epoch: int = 0,
    _callback_dict: dict = None,
    **kwargs,
):
    """
    Generates a sample from the diffusion model.
    """
    logger.info("Generating sample from diffusion model...")
    model.eval()
    kwargs = {} if kwargs is None else kwargs
    dataset_cfg: DatasetConfiguration = kwargs.get("dataset_cfg", cfg.datasets)
    spatial_size = kwargs.get(
        "spatial_size", dataset_cfg["transforms"]["load"][-1]["spatial_size"]
    )

    ## Sampling image during training
    image = torch.randn((1, 1, *spatial_size), device=device)
    image = image.to(device)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    with autocast(enabled=cfg.job.use_autocast):
        image = inferer.sample(
            input_noise=image, diffusion_model=model, scheduler=scheduler
        )

    plt.figure(figsize=(2, 2))
    plt.imshow(
        image[0, 0, :, :, spatial_size[-1] // 2].cpu(), vmin=0, vmax=1, cmap="gray"
    )
    plt.tight_layout()
    plt.axis("off")

    # return image
    if _callback_dict is not None:
        epoch = epoch + 1
        score_name = cfg.ignite.score_name
        sample_img_path = os.path.join(
            os.cwd(),
            "artifacts",
            "samples",
            f"sample_{epoch}_{score_name}={val_score}.png",
        )
        os.makedirs(os.path.dirname(sample_img_path), exist_ok=True)
        _callback_dict[score_name] = val_score
        plt.savefig(
            sample_img_path,
            bbox_inches="tight",
        )
        logger.info(f"Sample generated and saved to location '{sample_img_path}'.")
    plt.close()


def add_handlers(
    ignite_cfg: IgniteConfiguration,
    trainer: Engine,
    val_evaluator: Engine,
    optimizer: torch.optim.Optimizer = None,
    model: nn.Module = None,
):
    logger.info("Adding additional handlers...")
    score_name = ignite_cfg.score_name
    score_sign = -1.0 if score_name == "loss" else 1.0
    score_fn: callable = Checkpoint.get_default_score_fn(score_name, score_sign)
    handlers = {}

    if ignite_cfg.use_checkpoint:
        logger.info("Adding checkpoint for model...")
        global_step_transform = global_step_from_engine(trainer)
        checkpoint_kwargs = ignite_cfg.checkpoint
        to_save = {"model": model}
        checkpoint_handler: Checkpoint = hydra.utils.instantiate(
            ignite_cfg.checkpoint,
            global_step_transform=global_step_transform,
            score_function=score_fn,
            score_name=score_name,
            to_save=to_save,
            **checkpoint_kwargs,
        )
        val_evaluator.add_event_handler(
            Events.COMPLETED,
            handler=checkpoint_handler,
        )
        handlers["checkpoint"] = checkpoint_handler

    if ignite_cfg.use_early_stopping:
        logger.info("Adding early stopping...")
        early_stopping_kwargs = ignite_cfg.early_stopping
        early_stopping_handler: EarlyStopping = hydra.utils.instantiate(
            ignite_cfg.early_stopping,
            score_function=score_fn,
            trainer=trainer,
            **early_stopping_kwargs,
        )
        val_evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            handler=early_stopping_handler,
        )
        handlers["early_stopping"] = early_stopping_handler

    if ignite_cfg.use_lr_scheduler:
        logger.info("Adding learning rate scheduler...")
        lr_scheduler_kwargs = ignite_cfg.lr_scheduler
        lr_scheduler = create_lr_scheduler(
            optimizer, ignite_config=ignite_cfg, **lr_scheduler_kwargs
        )
        lr_scheduler_handler = LRScheduler(lr_scheduler)
        val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, lr_scheduler_handler)
        handlers["lr_scheduler"] = lr_scheduler_handler

    logger.info("Additional handlers added.\n")
    return handlers


def _sigmoid_output_transform(output):
    y_pred, y = output
    y_pred = torch.sigmoid(y_pred)
    return y_pred, y


# class SKLearnMetric(Metric):
#     def __init__(self, metric_fn: callable, output_transform=lambda x: x, device="cpu"):
#         super().__init__(output_transform, device)
#         self.metric_fn = metric_fn
#         self.metric_name: str = metric_fn.__name__


#     @sync_all_reduce("_num_examples", "_num_correct:SUM")
#     def compute(self):
#         if self._num_examples == 0:
#             raise NotComputableError(f'{self.metric_name} must have at least one example before it can be computed.')
#         return self._num_correct.item() / self._num_examples


def add_sklearn_metrics(
    old_metrics: dict,
    metrics: list,
):
    _ret_metrics = []
    for target_metric in metrics:
        metric_name = target_metric["_target_"].split(".")[-1]
        logger.debug("Adding: '{}'".format(metric_name))
        sklearn_metric_fn: callable = hydra_instantiate(target_metric, _partial_=True)
        sklearn_ignite_metric_fn = ignite_metrics_module.EpochMetric(
            compute_fn=sklearn_metric_fn, device=torch.device("cpu")
        )
        old_metrics[metric_name.lower()] = sklearn_ignite_metric_fn
        _ret_metrics.append(sklearn_metric_fn)

    return _ret_metrics


def create_metrics(
    cfg: Configuration = None,
    _metrics: dict = None,
    criterion: nn.Module = None,
    device: torch.device = torch.device("cpu"),
):
    """
    Creates torch ignite metrics for the model.
    ## Args:
        `ignite_config` (`IgniteConfiguration`, optional): The ignite configuration object. Defaults to `None`.
        `device` (`torch.device`, optional): The torch device to use. Defaults to `torch.device("cpu")`.
    ## Returns:
        `dict`: A dictionary of torch ignite metrics.
    """
    logger.info("Creating metrics...")
    ignite_config = cfg.ignite
    _metrics = _metrics if _metrics is not None else ignite_config.metrics
    metrics = dict()
    for metric_name, metric_fn_kwargs in _metrics.items():
        logger.info(f"Creating metric '{metric_name}'...")
        flag = False
        for mod in IGNITE_METRICS_MODULE:
            if hasattr(mod, metric_name):
                flag = True
                metric_fn_kwargs = {} if metric_fn_kwargs is None else metric_fn_kwargs
                metric_fn = getattr(mod, metric_name)
                if metric_name == "Loss":
                    metric_fn_kwargs["loss_fn"] = criterion
                # if metric_name == "ROC_AUC":
                #     metric_fn_kwargs["output_transform"] = _sigmoid_output_transform
                metrics[metric_name.lower()] = (
                    metric_fn(device=device, **metric_fn_kwargs)
                    if any(metric_fn_kwargs)
                    else metric_fn()
                )
        if metric_name == "Predictions":
            # create a custom metric to get predictions
            def __get_predictions(y_preds, y_true):
                y_preds = torch.argmax(y_preds, dim=1).cpu().numpy().astype(dtype=int)
                y_true = y_true.cpu().numpy()
                pred_dict = {"y_preds": y_preds, "y_true": y_true}
                return pred_dict

            metrics["predictions"] = ignite_metrics_module.EpochMetric(
                compute_fn=__get_predictions
            )
        elif not flag:
            logger.warn(f"Metric '{metric_name}' not found.")

    # if "metrics" in cfg.sklearn.keys() and any(cfg.sklearn.metrics):
    #     logger.info("Adding sklearn metrics...")
    #     sklearn_metrics = add_sklearn_metrics(
    #         old_metrics=metrics, metrics=cfg.sklearn.metrics
    #     )
    #     logger.debug(f"Sklearn metrics:\n{sklearn_metrics}\n")

    logger.info("Metrics created.\n")
    logger.debug(f"Metrics:\n{metrics}\n")

    return metrics


# def instantiate_ignite_metrics(ignite_cfg: IgniteConfiguration, **kwargs):
#     ignite_cfg = cfg.ignite


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    ignite_config: IgniteConfiguration,
    **lr_scheduler_kwargs,
):
    """
    Creates a LR scheduler for the model.

    ## Args:
        `optimizer` (`torch.optim.Optimizer`): The optimizer to use.
        `ignite_config` (`IgniteConfiguration`): The ignite configuration object.

    ## Returns:
        `_LRScheduler`: The LR scheduler instance.
    """
    logger.info("Creating LR scheduler...")
    lr_scheduler: torch_lr_schedulers._LRScheduler
    lr_scheduler = hydra.utils.instantiate(
        ignite_config.lr_scheduler, optimizer=optimizer, **lr_scheduler_kwargs
    )
    logger.info("LR scheduler created. LR scheduler summary:\n")
    inspect(lr_scheduler)

    return lr_scheduler


def _log_metrics_to_mlflow(metrics: dict, split: str, epoch: int):
    """Iterates through the metrics dictionary and logs the metrics to MLflow."""
    split_key = split + "_"
    logged_metrics = {}
    for key in metrics.keys():
        metric = metrics[key]
        if isinstance(metric, float):
            key_name = "".join([split_key, key])
            logged_metrics[key_name] = metric
    mlflow.log_metrics(logged_metrics, step=epoch)


def prepare_run(cfg: Configuration, loaders: dict, device: torch.device, **kwargs):
    logger.info("Preparing ignite run...")
    ignite_cfg: IgniteConfiguration = cfg.ignite

    ## prepare run
    trainer_args = create_default_trainer_args(cfg)
    trainer = create_supervised_trainer(**trainer_args)
    ProgressBar().attach(trainer)

    metrics = create_metrics(cfg, criterion=trainer_args["loss_fn"], device=device)
    # TODO: make the engines from monai.engines
    ## create evaluators
    train_evaluator = create_supervised_evaluator(
        trainer_args["model"], metrics=metrics, device=device
    )
    ProgressBar().attach(train_evaluator)
    val_evaluator = create_supervised_evaluator(
        trainer_args["model"], metrics=metrics, device=device
    )
    ProgressBar().attach(val_evaluator)
    test_evaluator = create_supervised_evaluator(
        trainer_args["model"], metrics=metrics, device=device
    )
    ProgressBar().attach(test_evaluator)

    # add additional handlers
    _ = add_handlers(
        ignite_cfg,
        trainer=trainer,
        model=trainer_args["model"],
        optimizer=trainer_args["optimizer"],
        val_evaluator=val_evaluator,
    )

    log_interval = ignite_cfg.get("log_interval", max(cfg.job.max_epochs // 10, 1))

    os.makedirs("artifacts/train/", exist_ok=True)
    os.makedirs("artifacts/val/", exist_ok=True)
    os.makedirs("artifacts/test/", exist_ok=True)

    def _log_metrics(evaluator: Engine, loader: DataLoader, split: str):
        evaluator.run(loader)
        metrics = evaluator.state.metrics
        epoch = trainer.state.epoch

        logger.info(
            f"epoch: {epoch}, {split} {ignite_cfg.score_name}: {metrics[ignite_cfg.score_name]}"
        )
        y_true = metrics["y_true"]
        y_pred = metrics["y_preds"]
        labels = cfg.datasets.labels

        # classification report
        cr_str = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=labels,
            zero_division=0.0,
        )
        logger.info(f"{split} classification report:\n{cr_str}")
        cr = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=labels,
            output_dict=True,
            zero_division=0.0,
        )
        cr_df = pd.DataFrame.from_dict(cr)
        cr_df.to_csv(f"artifacts/{split}/classification_report_epoch={epoch}.csv")
        # confusion matrix
        cfm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        cfm_df = pd.DataFrame(cfm, index=labels, columns=labels)
        logger.info(f"{split} confusion matrix:\n{cfm_df}")
        cfm_df.to_csv(f"artifacts/{split}/confusion_matrix_epoch={epoch}.csv")

        # roc_auc
        roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        metrics["roc_auc"] = roc_auc

        if cfg.job.use_mlflow:
            _log_metrics_to_mlflow(metrics=metrics, split=split, epoch=epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=log_interval))
    def log_metrics(trainer):
        logger.info("Logging metrics...")
        train_loader, val_loader = loaders[0], loaders[1]
        if not cfg.job.dry_run:
            _log_metrics(train_evaluator, train_loader, "train")
        else:
            logger.debug("Dry run, skipping train evaluation.")

        _log_metrics(val_evaluator, val_loader, "val")

    @trainer.on(Events.COMPLETED)
    def log_test_metrics(trainer):
        logger.info("Logging test metrics...")
        test_loader = loaders[2]
        _log_metrics(test_evaluator, test_loader, "test")

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception_raised(error: Exception):
        mlflow.end_run()

    return trainer, (train_evaluator, val_evaluator, test_evaluator)


def create_diffusion_model_engines(
    cfg: Configuration,
    loaders: list,
    trainer_args: dict,
    device: torch.device,
    **kwargs,
):
    job_cfg: JobConfiguration = cfg.job
    epoch_length: int = job_cfg.epoch_length
    max_epochs = job_cfg.max_epochs
    ignite_cfg: IgniteConfiguration = cfg.ignite
    val_interval: int = ignite_cfg.get("log_interval", max(job_cfg.max_epochs // 10, 1))
    model_cfg: DiffusionModelConfiguration = cfg.models
    num_train_timesteps: int = model_cfg.scheduler.num_train_timesteps
    val_handlers = [StatsHandler(name="train_log", output_transform=lambda x: None)]

    train_loader, val_loader = loaders[0], loaders[1]
    model = trainer_args["model"]
    optimizer = trainer_args["optimizer"]
    scheduler = models.instantiate_diffusion_scheduler(model_cfg)
    inferer = models.instantiate_diffusion_inferer(model_cfg, scheduler=scheduler)
    condition_name = "class"

    # TODO: Add validation metrics, particularly FID and SSIM
    evaluator = SupervisedEvaluator(
        device=device,
        epoch_length=epoch_length,
        inferer=inferer,
        network=model,
        prepare_batch=DiffusionPrepareBatch(
            num_train_timesteps=num_train_timesteps, condition_name=condition_name
        ),
        val_data_loader=val_loader,
        val_handlers=val_handlers,
        # additional_metrics=create_metrics(cfg, device=device),
        key_val_metric={
            "val_mean_abs_error": MeanAbsoluteError(
                output_transform=from_engine(["pred", "label"])
            )
        },
    )
    ProgressBar().attach(evaluator)

    train_handlers = [
        ValidationHandler(validator=evaluator, interval=val_interval, epoch_level=True),
        # StatsHandler(name="train_log", tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
    ]

    trainer = SupervisedTrainer(
        device=device,
        epoch_length=epoch_length,
        inferer=inferer,
        loss_function=trainer_args["loss_fn"],
        max_epochs=max_epochs,
        network=model,
        optimizer=optimizer,
        prepare_batch=DiffusionPrepareBatch(
            num_train_timesteps=num_train_timesteps, condition_name=condition_name
        ),
        train_data_loader=train_loader,
        train_handlers=train_handlers,
        key_train_metric={
            "train_accuracy": MeanSquaredError(
                output_transform=from_engine(["pred", "label"])
            )
        },
    )
    ProgressBar().attach(trainer)

    return trainer, evaluator


def prepare_diffusion_run(
    cfg: Configuration,
    loaders: dict,
    device: torch.device,
    **kwargs,
):
    logger.info("Preparing ignite run...")
    ignite_cfg = cfg.ignite
    trainer_args = create_default_trainer_args(cfg)

    ## prepare run
    trainer, evaluator = create_diffusion_model_engines(
        cfg, loaders, trainer_args=trainer_args, device=device, **kwargs
    )

    # logger.info("Running diffusion model evaluator...")
    # state = evaluator.run(loader)
    ## create evaluators
    # metrics = create_metrics(cfg, criterion=trainer_args["loss_fn"], device=device)
    # evaluator = create_supervised_evaluator(
    #     trainer_args["model"], metrics=metrics, device=device
    # )
    # score_name = cfg.ignite.score_name
    # sample_from_diffusion_model_kwargs["epoch"] = state.epoch
    # sample_from_diffusion_model_kwargs["val_score"] = state.output.cpu().numpy()
    # sample_from_diffusion_model(cfg=cfg, **sample_from_diffusion_model_kwargs)
    # logger.debug(f"State:\n{state}\n")

    # # TODO: Callback for MLflow
    log_interval = ignite_cfg.get("log_interval", max(cfg.job.max_epochs // 10, 1))

    def _log_metrics(evaluator: Engine, loader: DataLoader, split: str):
        evaluator.run(loader)
        metrics = evaluator.state.metrics
        epoch = trainer.state.epoch

        # logger.info(
        #     f"epoch: {epoch}, {split} {ignite_cfg.score_name}: {metrics[ignite_cfg.score_name]}"
        # )
        logger.info(f"epoch: {epoch}\n{metrics}")
        # y_true = metrics["y_true"]
        # y_pred = metrics["y_preds"]
        # labels = cfg.datasets.labels

        # # classification report
        # cr_str = classification_report(
        #     y_true=y_true,
        #     y_pred=y_pred,
        #     target_names=labels,
        #     zero_division=0.0,
        # )
        # logger.info(f"{split} classification report:\n{cr_str}")
        # cr = classification_report(
        #     y_true=y_true,
        #     y_pred=y_pred,
        #     target_names=labels,
        #     output_dict=True,
        #     zero_division=0.0,
        # )
        # cr_df = pd.DataFrame.from_dict(cr)
        # cr_df.to_csv(f"artifacts/{split}/classification_report_epoch={epoch}.csv")
        # # confusion matrix
        # cfm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        # cfm_df = pd.DataFrame(cfm, index=labels, columns=labels)
        # logger.info(f"{split} confusion matrix:\n{cfm_df}")
        # cfm_df.to_csv(f"artifacts/{split}/confusion_matrix_epoch={epoch}.csv")

        # # roc_auc
        # roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        # metrics["roc_auc"] = roc_auc

        if cfg.job.use_mlflow:
            _log_metrics_to_mlflow(metrics=metrics, split=split, epoch=epoch)

    # @trainer.on(Events.EPOCH_COMPLETED(every=log_interval))
    # def log_metrics(trainer):
    #     logger.info("Logging metrics...")
    #     train_loader, val_loader = loaders[0], loaders[1]
    #     if not cfg.job.dry_run:
    #         _log_metrics(train_evaluator, train_loader, "train")
    #     else:
    #         logger.debug("Dry run, skipping train evaluation.")

    #     _log_metrics(val_evaluator, val_loader, "val")

    # @trainer.on(Events.EPOCH_COMPLETED(every=log_interval))
    # def log_metrics(trainer):
    #     logger.info("Logging metrics...")
    #     if not cfg.job.dry_run:
    #         _log_metrics(train_evaluator, train_loader, "train")
    #     else:
    #         logger.debug("Dry run, skipping train evaluation.")

    #     _log_metrics(val_evaluator, val_loader, "val")

    return trainer, evaluator
