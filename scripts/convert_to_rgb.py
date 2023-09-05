#!/usr/bin/env python
"""Converts all images in a given directory to RGB format."""
import argparse
import cv2
import os
import sys
from omegaconf import OmegaConf, DictConfig

# sys path append
CWD = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(CWD)))

# ttk
from ttk.utils import get_logger

CHEST_XRAY_IMG_PATH = "/home/nicoleg/workspaces/teddytoolkit/ttk/.cache/datasets/scans/Chest_XRay_Images:latest/"
CHEST_XRAY_RGB_PATH = (
    "/home/nicoleg/workspaces/teddytoolkit/.data/Chest_XRay_Images_RGB/"
)

logger = get_logger("convert_to_rgb")


def convert(input_dir: str, output_dir: str, dry_run: bool = False):
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpeg"):
            img = cv2.imread(os.path.join(input_dir, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not dry_run:
                cv2.imwrite(os.path.join(output_dir, filename), img)
            logger.info(
                f"Converted {os.path.join(output_dir, filename)} to RGB format. Saved to {output_dir}."
            )


def main(cfg: DictConfig):
    ## input configuration setup
    input_train_normal_dir = os.path.join(cfg.input_dir, "train", "NORMAL")
    input_train_pneumonia_dir = os.path.join(cfg.input_dir, "train", "PNEUMONIA")
    input_test_normal_dir = os.path.join(cfg.input_dir, "test", "NORMAL")
    input_test_pneumonia_dir = os.path.join(cfg.input_dir, "test", "PNEUMONIA")

    ## output configuration setup
    output_train_normal_dir = os.path.join(cfg.output_dir, "train", "NORMAL")
    output_train_pneumonia_dir = os.path.join(cfg.output_dir, "train", "PNEUMONIA")
    output_test_normal_dir = os.path.join(cfg.output_dir, "test", "NORMAL")
    output_test_pneumonia_dir = os.path.join(cfg.output_dir, "test", "PNEUMONIA")

    ## create output directories
    os.makedirs(output_train_normal_dir, exist_ok=True)
    os.makedirs(output_train_pneumonia_dir, exist_ok=True)
    os.makedirs(output_test_normal_dir, exist_ok=True)
    os.makedirs(output_test_pneumonia_dir, exist_ok=True)

    ## convert to RGB
    convert(input_train_normal_dir, output_train_normal_dir, cfg.dry_run)
    convert(input_train_pneumonia_dir, output_train_pneumonia_dir, cfg.dry_run)
    convert(input_test_normal_dir, output_test_normal_dir, cfg.dry_run)
    convert(input_test_pneumonia_dir, output_test_pneumonia_dir, cfg.dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=CHEST_XRAY_IMG_PATH,
        help="Directory containing images to convert.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=CHEST_XRAY_RGB_PATH,
        help="Directory to save converted images to.",
    )
    parser.add_argument(
        "--dry_run",
        type=bool,
        default=False,
        help="If True, does not save images to output directory.",
    )
    args = parser.parse_args()
    cfg = OmegaConf.create(vars(args))
    main(cfg)
