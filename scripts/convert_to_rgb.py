#!/usr/bin/env python
"""Converts all images in a given directory to RGB format."""
import argparse
import cv2
import os
import pandas as pd
import sys
from omegaconf import OmegaConf, DictConfig

# rtk
from rtk.utils import get_logger

CHEST_XRAY_IMG_PATH = (
    "/home/nicoleg/workspaces/teddytoolkit/.data/Chest_XRay_14_Kaggle/"
)
CHEST_XRAY_RGB_PATH = (
    "/home/nicoleg/workspaces/teddytoolkit/.data/Chest_XRay_14_Kaggle_RGB/"
)

logger = get_logger("convert_to_rgb")


def convert(
    input_dir: os.PathLike = CHEST_XRAY_IMG_PATH,
    output_dir: os.PathLike = CHEST_XRAY_RGB_PATH,
    image_files: list = None,
    dry_run: bool = False,
    version: float = 1.0,
):
    if version == 1.0:
        for filename in os.listdir(input_dir):
            if filename.endswith(".jpeg") or filename.endswith(".png"):
                img = cv2.imread(os.path.join(input_dir, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if not dry_run:
                    cv2.imwrite(os.path.join(output_dir, filename), img)
                logger.info(
                    f"Converted {os.path.join(input_dir, filename)} to RGB format. Saved to {os.path.join(output_dir, filename)}."
                )
    elif version == 2.0:
        for filepath in image_files:
            filename = os.path.basename(filepath)
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not dry_run:
                cv2.imwrite(os.path.join(output_dir, filename), img)
            logger.info(
                f"Converted {filepath} to RGB format. Saved to {os.path.join(output_dir, filename)}."
            )


def unpack_images(cfg: DictConfig):
    """"""
    scan_path: os.PathLike = os.path.join(cfg.input_dir)
    image_files = []

    for encoding in range(1, 13):
        label_path = os.path.join(scan_path, f"images_{encoding:03}", "images")

        for filename in os.listdir(label_path):
            image_files.append(os.path.join(label_path, filename))

    return image_files


def main(cfg: DictConfig, unpack=True):
    os.makedirs(cfg.output_dir, exist_ok=True)
    if unpack:
        input_dir: os.PathLike = cfg.input_dir
        output_dir: os.PathLike = cfg.output_dir
        image_files = unpack_images(cfg)
        convert(
            input_dir,
            output_dir,
            image_files=image_files,
            dry_run=cfg.dry_run,
            version=2.0,
        )

    else:
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
    main(cfg, unpack=True)
