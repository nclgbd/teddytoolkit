# `ttk.configs` README.md

## Hydra settings

```yaml
# python scripts/run_train.py --config-dir=$(pwd)/configs/ --config-name=CONFIG_NAME

defaults:
  - _self_
  - datasets: ???
  - job: test
  - models: resnet-model

  # module configurations
  - ignite: ignite
  - mlflow: local
  - sklearn: classifier

  # overrides:
  - override hydra/job_logging: colorlog
  - override hydra/hydra_logging: colorlog

# hydra settings
hydra:
  job:
    chdir: true
  sweep:
    dir: outputs/${hydra.job.config_name}/${date}/${timestamp}
    subdir: ${hydra.job.override_dirname}
  run:
    dir: outputs/${hydra.job.config_name}/${date}/${timestamp}
```
