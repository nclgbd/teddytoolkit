from omegaconf import OmegaConf

from ttk.config import Configuration


class TestConfig:
    def test_config(self, test_cfg: Configuration):
        assert test_cfg is not None
        # dataset checks
        assert test_cfg.dataset is not None
        # job checks
        assert test_cfg.job is not None
        assert test_cfg.job.debug == True
        assert test_cfg.job.dry_run == True
        # model checks
