# imports
import pytest
import os

# azureml
from azureml.core import Workspace

# rtk
from rtk import utils

logger = utils.get_logger("test_logger")


class TestUtilities:
    @pytest.fixture
    def ws(self):
        """The AzureML workspace object."""
        return utils.login()

    def test_load_patient_dataset(self, ws: Workspace):
        """Test the `rtk.load_patient_dataset` function."""
        dataset = utils.load_patient_dataset(
            ws, patient_dataset_name="Brain_Tumor_MRI_metadata"
        )
        assert dataset is not None

    def test_load_scan_dataset(self, ws: Workspace):
        """Test the `rtk.load_scan_dataset` function."""
        # with mount=True
        dataset_mount = utils.load_scan_dataset(
            ws, scan_dataset_name="Brain_Tumor_MRI_DICOM", mount=True
        )
        assert dataset_mount is not None
        data_dir = dataset_mount.mount_point
        dataset_mount.start()
        files = os.listdir(data_dir)
        assert any(files)
        dataset_mount.stop()
        # with mount=False
        dataset_mount = utils.load_scan_dataset(
            ws, scan_dataset_name="Brain_Tumor_MRI_DICOM", mount=False
        )
