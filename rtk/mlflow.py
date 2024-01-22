#
from copy import deepcopy
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# mlflow
import mlflow

# rtk
from rtk.utils import get_logger, _strip_target, login
from rtk.config import *

_logger = get_logger(__name__)


def create_run_name(cfg: Configuration, random_state: int, **kwargs):
    """Create a run name."""
    dataset_cfg = cfg.datasets
    preprocessing_cfg = dataset_cfg.preprocessing
    job_cfg = cfg.job
    tags = job_cfg.get("tags", {})
    model_cfg = cfg.models

    model_name = model_cfg.model._target_.split(".")[-1]
    run_name: str = model_cfg.model.get("model_name", model_name.lower())

    if tags.get("type", "train") == "train" or job_cfg.mode == "train":
        criterion_name: str = model_cfg.criterion._target_.split(".")[-1].lower()
        lr: float = model_cfg.optimizer.lr
        optimizer_name: str = model_cfg.optimizer._target_.split(".")[-1].lower()
        weight_decay: float = model_cfg.optimizer.get("weight_decay", 0.0)
        run_name += f";optimizer={optimizer_name};lr={lr};weight_decay={weight_decay};criterion={criterion_name}"

        if preprocessing_cfg.use_sampling:
            sample_to_value = preprocessing_cfg.sampling_method["sample_to_value"]
            run_name += f";sample_to_value={sample_to_value}"

        run_name += f";pretrained={str(job_cfg.use_pretrained).lower()}"

    elif tags.get("type", "train") == "eval" or job_cfg.mode == "evaluate":
        pretrained_model = model_cfg.get("load_model", {}).get("name", "")
        run_name += f";pretrained_model={pretrained_model}"

    elif tags.get("type", "train") == "diff" or job_cfg.mode == "diffusion":
        pass

    date = cfg.date
    postfix: str = cfg.get("postfix", "")
    timestamp = cfg.timestamp
    run_name = "".join(
        [run_name, f";seed={random_state}", f";{date}-{timestamp}", f";{postfix}"]
    )
    return run_name


def get_params(cfg: Configuration, **kwargs):
    """
    Get the parameters of this run.
    """
    dataset_cfg: DatasetConfiguration = kwargs.get("dataset_cfg", cfg.datasets)
    job_cfg: JobConfiguration = kwargs.get("job_cfg", cfg.job)
    model_cfg: ModelConfiguration = kwargs.get("model_cfg", cfg.models)
    preprocessing_cfg: PreprocessingConfiguration = kwargs.get(
        "preprocessing_cfg", dataset_cfg.preprocessing
    )

    params = dict()

    # NOTE: job parameters
    def __collect_job_params():
        params["max_epochs"] = job_cfg.max_epochs
        params["use_pretrained"] = job_cfg.use_pretrained
        params["use_transforms"] = job_cfg.use_transforms

    __collect_job_params()

    # NOTE: model parameters
    def __collect_model_params():
        # model parameters
        params["model_name"] = model_cfg.model.get(
            "model_name", _strip_target(model_cfg.model, lower=True)
        )
        params.update(model_cfg.model)
        # criterion parameters
        params["criterion_name"] = params.get(
            "criterion", _strip_target(model_cfg.criterion)
        )
        params.update(model_cfg.criterion)
        # optimizer parameters
        params["optimizer_name"] = params.get(
            "optimizer", _strip_target(model_cfg.optimizer)
        )
        params.update(model_cfg.optimizer)

    __collect_model_params()

    # NOTE: dataset parameters
    def __collect_dataset_params():
        params["dataset_name"] = dataset_cfg.name
        params.update(preprocessing_cfg)
        if params["use_sampling"] == False:
            del params["sampling_method"]

    __collect_dataset_params()

    # in case '_target_' somehow wasn't found
    try:
        del params["_target_"]
    except KeyError:
        pass

    return params


def log_mlflow_params(cfg: Configuration, **kwargs):
    """
    Log the parameters to MLFlow.
    """
    tags = cfg.job.get("tags", {})
    params = get_params(cfg, **kwargs)
    logger.info("Logged parameters:\n{}".format(OmegaConf.to_yaml(params)))
    mlflow.log_params(params)
    if any(tags):
        mlflow.set_tags(tags)


def prepare_mlflow(cfg: Configuration):
    logger.info("Starting MLflow run...")
    mlflow_cfg = cfg.mlflow
    if cfg.job.use_azureml:
        logger.debug("Using AzureML for experiment tracking...")
        ws = login()
        tracking_uri = ws.get_mlflow_tracking_uri()

    else:
        tracking_uri = mlflow_cfg.get("tracking_uri", "~/mlruns/")

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = mlflow_cfg.get(
        "experiment_name", HydraConfig.get().job.config_name
    )
    experiment_id = mlflow.create_experiment(
        experiment_name, artifact_location=tracking_uri
    )
    logger.debug(f"MLflow tracking URI: {tracking_uri}")
    start_run_kwargs: dict = mlflow_cfg.get("start_run", {})
    start_run_kwargs["experiment_id"] = experiment_id
    return start_run_kwargs
