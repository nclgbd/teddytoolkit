"""
Tests for the `rtk.config` module.
"""
from rtk.config import Configuration


class TestConfig:
    def test_config(self, test_cfg: Configuration):
        assert test_cfg is not None
        # dataset checks
        assert test_cfg.datasets is not None
        # job checks
        assert test_cfg.job is not None
        assert test_cfg.job.dry_run == True
        # datasets
        assert test_cfg.datasets is not None
        assert any(test_cfg.datasets.dataloader._target_)
        assert any(test_cfg.datasets.instantiate._target_)
        # model checks
        assert any(test_cfg.models.model._target_)
        assert any(test_cfg.models.criterion._target_)
        assert any(test_cfg.models.optimizer._target_)

        ## module configuration checks
        assert any(test_cfg.ignite)
        assert any(test_cfg.mlflow)
        assert any(test_cfg.sklearn)
