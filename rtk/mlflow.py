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


def _determine_model_name(cfg: Configuration, **kwargs):
    model_cfg = cfg.models
    model_name: str = _strip_target(model_cfg.model, lower=True)
    if model_name == "from_pretrained":
        model_name = model_cfg.model.pretrained_model_name_or_path.split("/")[-1]

    return model_name


def create_run_name(cfg: Configuration, random_state: int, **kwargs):
    """Create a run name."""
    dataset_cfg = cfg.datasets
    preprocessing_cfg = dataset_cfg.preprocessing
    job_cfg = cfg.job
    tags = job_cfg.get("tags", {})
    model_cfg = cfg.models

    model_name = _determine_model_name(cfg, **kwargs)
    run_name: str = model_cfg.model.get("model_name", model_name.lower())

    if tags.get("type", "train") == "train" or cfg.mode == "train":
        criterion_name: str = model_cfg.criterion._target_.split(".")[-1].lower()
        lr: float = model_cfg.optimizer.lr
        optimizer_name: str = model_cfg.optimizer._target_.split(".")[-1].lower()
        weight_decay: float = model_cfg.optimizer.get("weight_decay", 0.0)
        run_name += f";optimizer={optimizer_name};lr={lr};weight_decay={weight_decay};criterion={criterion_name}"

        if preprocessing_cfg.use_sampling:
            sample_to_value = preprocessing_cfg.sampling_method["sample_to_value"]
            run_name += f";sample_to_value={sample_to_value}"

        run_name += f";pretrained={str(job_cfg.use_pretrained).lower()}"

    elif tags.get("type", "train") == "eval" or cfg.mode == "evaluate":
        pretrained_model = model_cfg.get("load_model", {}).get("name", "")
        run_name += f";pretrained_model={pretrained_model}"

    elif tags.get("type", "train") == "diff" or cfg.mode == "diffusion":
        pass

    date = cfg.date
    postfix: str = cfg.postfix
    timestamp = cfg.timestamp
    run_name = "".join(
        [run_name, f";seed={random_state}", f";{date}-{timestamp}", f";{postfix}"]
    )
    return run_name


def get_base_params(cfg: Configuration, **kwargs):
    params = dict()
    params["date"] = cfg.date
    params["postfix"] = cfg.postfix
    params["random_state"] = cfg.random_state
    params["timestamp"] = cfg.timestamp
    params["use_transforms"] = cfg.use_transforms
    return params


def get_params(cfg: Configuration, **kwargs):
    """
    Get the parameters of this run.
    """
    dataset_cfg: DatasetConfiguration = kwargs.get("dataset_cfg", cfg.datasets)
    model_cfg: ModelConfiguration = kwargs.get("model_cfg", cfg.models)
    preprocessing_cfg: PreprocessingConfiguration = kwargs.get(
        "preprocessing_cfg", dataset_cfg.preprocessing
    )

    # NOTE: cfg parameters
    params = get_base_params(cfg, **kwargs)

    # NOTE: model parameters
    def __collect_model_params():
        # model parameters
        model_name = _determine_model_name(cfg, **kwargs)
        params["model_name"] = model_name
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
    # tags = cfg.get("tags", {})
    params = get_params(cfg, **kwargs)
    logger.info("Logged parameters:\n{}".format(OmegaConf.to_yaml(params)))
    mlflow.log_params(params)


def prepare_mlflow(cfg: Configuration):
    logger.info("Preparing MLflow run...")
    mlflow_cfg = cfg.mlflow
    logger.debug("Using AzureML for experiment tracking...")
    ws = login()
    tracking_uri = ws.get_mlflow_tracking_uri()

    mlflow.set_tracking_uri(tracking_uri)

    try:
        experiment_name = mlflow_cfg.get(
            "experiment_name", HydraConfig.get().job.config_name
        )
    except ValueError:
        experiment_name = mlflow_cfg.get("experiment_name", "Default")
    experiment_id = mlflow.create_experiment(
        experiment_name, artifact_location=tracking_uri
    )
    logger.debug(f"MLflow tracking URI: {tracking_uri}")
    start_run_kwargs: dict = mlflow_cfg.get("start_run", {})
    start_run_kwargs["experiment_id"] = experiment_id
    return start_run_kwargs
