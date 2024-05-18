# imports
from collections import Counter
from typing import Callable, Union
import hydra
import logging
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from omegaconf import OmegaConf
import pandas as pd
from rich import inspect
from hydra.core.hydra_config import HydraConfig

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
from rtk._datasets import LABEL_KEYNAME
from rtk.config import *
from rtk.utils import get_logger, _console, stringify_epoch, hydra_instantiate

logger = get_logger(__name__)
console = _console


def create_default_trainer_args(
    cfg: ImageConfiguration,
    **kwargs,
):
    """
    Prepares the data for the ignite trainer.
    """
    console.log("Creating default trainer arguments...")
    trainer_kwargs = dict()
    device: torch.device = kwargs.get("device", torch.device(cfg.device))
    trainer_kwargs["device"] = device

    # Prepare model, optimizer, loss function, and criterion
    model: nn.Module = models.instantiate_model(cfg, device=device)
    trainer_kwargs["model"] = model
    criterion_kwargs = kwargs.get("criterion_kwargs", {})
    criterion = models.instantiate_criterion(cfg, device=device, **criterion_kwargs)
    trainer_kwargs["loss_fn"] = criterion
    optimizer: torch.optim.Optimizer = models.instantiate_optimizer(cfg, model=model)
    trainer_kwargs["optimizer"] = optimizer

    return trainer_kwargs


def add_handlers(
    ignite_cfg: IgniteConfiguration,
    trainer: Engine,
    val_evaluator: Engine = None,
    optimizer: torch.optim.Optimizer = None,
    model: nn.Module = None,
):
    console.log("Adding additional handlers...")
    score_name = ignite_cfg.score_name
    score_sign = -1.0 if score_name == "loss" else 1.0
    score_fn: callable = Checkpoint.get_default_score_fn(score_name, score_sign)
    handlers = []

    if ignite_cfg.use_checkpoint:
        console.log("Adding checkpoint for model...")
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
        if isinstance(val_evaluator, Engine):
            val_evaluator.add_event_handler(
                Events.COMPLETED,
                handler=checkpoint_handler,
            )

        handlers.append(checkpoint_handler)

    if ignite_cfg.use_early_stopping:
        console.log("Adding early stopping...")
        early_stopping_kwargs = ignite_cfg.early_stopping
        early_stopping_handler: EarlyStopping = hydra.utils.instantiate(
            ignite_cfg.early_stopping,
            score_function=score_fn,
            trainer=trainer,
            **early_stopping_kwargs,
        )
        if isinstance(val_evaluator, Engine):
            val_evaluator.add_event_handler(
                Events.EPOCH_COMPLETED,
                handler=early_stopping_handler,
            )
        handlers.append(early_stopping_handler)

    if ignite_cfg.use_lr_scheduler:
        console.log("Adding learning rate scheduler...")
        lr_scheduler_kwargs = ignite_cfg.lr_scheduler
        lr_scheduler = create_lr_scheduler(
            optimizer, ignite_config=ignite_cfg, **lr_scheduler_kwargs
        )
        lr_scheduler_handler = LRScheduler(lr_scheduler)
        if isinstance(val_evaluator, Engine):
            val_evaluator.add_event_handler(
                Events.EPOCH_COMPLETED, lr_scheduler_handler
            )

        handlers.append(lr_scheduler_handler)

    console.log("Additional handlers added.\n")
    return handlers


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
    cfg: ImageClassificationConfiguration = None,
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
    console.log("Creating metrics...")
    ignite_config = cfg.ignite
    _metrics = _metrics if _metrics is not None else ignite_config.metrics
    metrics = dict()
    for metric_name, metric_fn_kwargs in _metrics.items():
        logger.debug(f"Creating metric '{metric_name}'...")
        flag = False
        for mod in IGNITE_METRICS_MODULE:
            if hasattr(mod, metric_name):
                flag = True
                metric_fn_kwargs = {} if metric_fn_kwargs is None else metric_fn_kwargs
                metric_fn = getattr(mod, metric_name)
                if metric_name == "Loss":
                    metric_fn_kwargs["loss_fn"] = criterion

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

    console.log("Metrics created.\n")
    logger.debug(f"Metrics:\n{metrics}\n")

    return metrics


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
    console.log("Creating LR scheduler...")
    lr_scheduler: torch_lr_schedulers._LRScheduler
    lr_scheduler = hydra.utils.instantiate(
        ignite_config.lr_scheduler, optimizer=optimizer, **lr_scheduler_kwargs
    )
    console.log("LR scheduler created.")
    logger.debug("LR scheduler summary:\n")
    inspect(lr_scheduler)

    return lr_scheduler


def build_report(
    cfg: ImageClassificationConfiguration,
    metrics: dict,
    epoch: int,
    split: str = "test",
    **kwargs,
):
    """
    Creates a report for the model.
    """
    console.log("Generating report...")
    model_cfg: ModelConfiguration = cfg.get(
        "models", kwargs.get("model_cfg", ModelConfiguration())
    )
    report: str = "# Run summary\n\n"
    epoch_str = stringify_epoch(epoch)
    # classification report
    cr_report_dict = dict()
    cr_df: pd.DataFrame = pd.read_csv(
        os.path.join("artifacts", split, f"classification_report_epoch={epoch_str}.csv")
    )

    model_class_name = model_cfg.model._target_.split(".")[-1].lower()
    model_name = model_cfg.model.get("model_name", model_class_name)
    test_auc: float = metrics[f"{split}_roc_auc"]
    test_acc: float = metrics[f"{split}_accuracy"]
    test_precision: float = cr_df["macro avg"][0]
    test_recall: float = cr_df["macro avg"][1]
    test_f1: float = cr_df["macro avg"][2]

    cr_report_dict["model_name"] = model_name
    cr_report_dict["roc_auc"] = test_auc
    cr_report_dict["accuracy"] = test_acc
    cr_report_dict["precision"] = test_precision
    cr_report_dict["recall"] = test_recall
    cr_report_dict["f1-score"] = test_f1
    cr_report_df = pd.DataFrame.from_dict(
        cr_report_dict,
        orient="index",
    )
    cr_report_df = cr_report_df.transpose().set_index("model_name")

    report += f"## Test results\n"
    report += f"### Evaluation metrics\n{cr_report_df.to_markdown()}\n\n"

    # confusion matrix
    cfm_df: pd.DataFrame = (
        pd.read_csv(
            os.path.join("artifacts", split, f"confusion_matrix_epoch={epoch_str}.csv")
        )
        .set_index("Unnamed: 0")
        .rename_axis("", axis=0)
    )
    report += f"### Confusion matrix\n{cfm_df.to_markdown()}\n\n"

    # configuration
    cfg_yaml = OmegaConf.to_yaml(cfg, sort_keys=True)
    report += f"## `config.yaml`\n```yaml\n# --config-name=config\n\n{cfg_yaml}\n```\n"

    with open("artifacts/report.md", "w") as f:
        f.write(report)

    if cfg.job.use_mlflow:
        mlflow.set_tag("mlflow.note.content", report)

    return report


def _log_metrics(
    cfg: ImageClassificationConfiguration,
    trainer: Engine,
    evaluator: Engine,
    loader: DataLoader,
    split: str,
    **kwargs,
):
    ignite_cfg: IgniteConfiguration = cfg.ignite
    evaluator.run(loader)
    metrics = evaluator.state.metrics
    epoch = trainer.state.epoch
    epoch_str = stringify_epoch(epoch)
    labels = sorted(list(cfg.datasets.encoding.keys()))

    console.print(f"'{split.capitalize()}' results for epoch: {epoch}\n")
    console.print(
        f"'{split}' {ignite_cfg.score_name}: {metrics[ignite_cfg.score_name]}"
    )
    y_true = metrics["y_true"]
    y_pred = metrics["y_preds"]

    # roc_auc
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    metrics["roc_auc"] = roc_auc
    console.print(f"'{split}' roc_auc: {roc_auc}")

    try:
        image_files = loader.dataset.image_files
        index = pd.Index([os.path.basename(image_file) for image_file in image_files])
        predictions_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=index)
        predictions_df.to_csv(f"artifacts/{split}/predictions_epoch={epoch_str}.csv")

    except AttributeError as e:
        logger.warn(f"Could not save predictions due to: {e}")

    # classification report
    cr_str = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=labels,
        zero_division=0.0,
    )
    console.print(f"'{split}' classification report:\n{cr_str}")
    cr = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0.0,
    )
    cr_df = pd.DataFrame.from_dict(cr)
    metrics["f1_score"] = cr_df["macro avg"][2]
    z_epoch_str = stringify_epoch(epoch)
    cr_filepath = f"artifacts/{split}/classification_report_epoch={z_epoch_str}.csv"
    cr_df.to_csv(cr_filepath)
    # confusion matrix
    cfm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cfm_df = pd.DataFrame(cfm, index=labels, columns=labels)
    console.print(f"'{split}' confusion matrix:\n{cfm_df}")
    cfm_filepath = f"artifacts/{split}/confusion_matrix_epoch={z_epoch_str}.csv"
    cfm_df.to_csv(cfm_filepath)

    if cfg.job.use_mlflow:
        console.log("Uploading metrics to MLflow...")
        override = kwargs.get("override", False)
        _log_metrics_to_mlflow(
            cfg, metrics=metrics, split=split, epoch=epoch, override=override
        )

        # log the physical classification and confusion matrices
        mlflow.log_artifact(cr_filepath, f"{split}/")
        mlflow.log_artifact(cfm_filepath, f"{split}/")


def _log_metrics_to_mlflow(
    cfg: ImageClassificationConfiguration,
    metrics: dict,
    split: str,
    epoch: int,
    generate_report: bool = True,
    override: bool = False,
):
    """Iterates through the metrics dictionary and logs the metrics to MLflow."""
    split_key = split + "_"
    logged_metrics = {}
    for key in metrics.keys():
        metric = metrics[key]
        if isinstance(metric, float):
            key_name = "".join([split_key, key])
            logged_metrics[key_name] = metric

    if (generate_report and split == "test") or override:
        report = build_report(cfg=cfg, metrics=logged_metrics, epoch=epoch, split=split)
        logger.debug(f"Report:\n{report}")

    mlflow.log_metrics(logged_metrics, step=epoch)


def train(
    cfg: ImageClassificationConfiguration,
    trainer: Engine,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loaders: list,
    metrics: dict,
    device: torch.device,
    **kwargs,
):
    ignite_cfg = cfg.ignite
    ## create evaluators
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    ProgressBar().attach(train_evaluator)
    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    ProgressBar().attach(val_evaluator)
    test_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    ProgressBar().attach(test_evaluator)

    # add additional handlers
    _ = add_handlers(
        ignite_cfg,
        trainer=trainer,
        model=model,
        optimizer=optimizer,
        val_evaluator=val_evaluator,
    )

    log_interval = ignite_cfg.get("log_interval", max(cfg.job.max_epochs // 10, 1))

    os.makedirs("artifacts/train/", exist_ok=True)
    os.makedirs("artifacts/val/", exist_ok=True)
    os.makedirs("artifacts/test/", exist_ok=True)

    @trainer.on(Events.EPOCH_COMPLETED(every=log_interval))
    def log_metrics(trainer):
        console.log("Logging metrics...")
        train_loader, val_loader = loaders[0], loaders[1]
        if not cfg.job.dry_run:
            _log_metrics(cfg, trainer, train_evaluator, train_loader, "train")
        else:
            logger.warn("Dry run, skipping train evaluation.")

        _log_metrics(cfg, trainer, val_evaluator, val_loader, "val")

    @trainer.on(Events.COMPLETED)
    def log_test_metrics(trainer):
        console.log("Logging test metrics...")
        test_loader = loaders[2]
        _log_metrics(cfg, trainer, test_evaluator, test_loader, "test")

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception_raised(error: Exception):
        logger.warn("Exception raised, ending run...")
        mlflow.end_run()

    return trainer, [train_evaluator, val_evaluator, test_evaluator]


def evaluate(
    cfg: ImageClassificationConfiguration,
    trainer: Engine,
    model: nn.Module,
    loader: DataLoader,
    metrics: dict,
    device: torch.device,
    **kwargs,
):
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    ProgressBar().attach(evaluator)
    _log_metrics(cfg, trainer, evaluator, loader, "test", override=True)


def prepare_run(
    cfg: ImageClassificationConfiguration,
    loaders: list,
    device: torch.device,
    mode: str = "train",
    **kwargs,
):
    console.log("Preparing ignite run...")

    ## prepare run
    default_trainer_kwargs = {}

    trainer_args = create_default_trainer_args(cfg, **default_trainer_kwargs)
    trainer = create_supervised_trainer(**trainer_args)
    ProgressBar().attach(trainer)

    metrics = create_metrics(cfg, criterion=trainer_args["loss_fn"], device=device)

    trainer: Engine
    evaluators: list
    model: nn.Module = trainer_args["model"]
    if mode == "train":
        trainer, evaluators = train(
            cfg=cfg,
            trainer=trainer,
            model=model,
            optimizer=trainer_args["optimizer"],
            loaders=loaders,
            metrics=metrics,
            device=device,
        )
    elif mode == "evaluate":
        # load weights
        model.eval()
        evaluate(
            cfg=cfg,
            trainer=trainer,
            model=model,
            loader=loaders[-1],
            metrics=metrics,
            device=device,
        )
        return

    else:
        raise ValueError(
            f"Invalid mode '{mode}'. Valid modes are 'train' and 'evaluate'."
        )

    return trainer, evaluators
