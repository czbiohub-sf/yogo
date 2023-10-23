#! /usr/bin/env python3

"""
simple script to act as cli for testing a yogo model
"""

import argparse

from pathlib import Path

from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split

from yogo.model import YOGO
from yogo.train import Trainer
from yogo.utils import choose_device, Timer
from yogo.data.yogo_dataset import ObjectDetectionDataset
from yogo.data.yogo_dataloader import choose_dataloader_num_workers
from yogo.utils.default_hyperparams import DefaultHyperparams as df
from yogo.data.dataset_description_file import load_dataset_description


def load_description_to_dataloader(
    dataset_description_file: Path, Sx: int, Sy: int, normalize_images: bool
):
    dataset_descriptor = load_dataset_description(str(dataset_description_file))
    all_dataset_paths = dataset_descriptor.dataset_paths + (
        dataset_descriptor.test_dataset_paths or []
    )

    dataset : ConcatDataset[ObjectDetectionDataset] = ConcatDataset(
        ObjectDetectionDataset(
            dsp["image_path"],
            dsp["label_path"],
            Sx,
            Sy,
            normalize_images=normalize_images,
        )
        for dsp in all_dataset_paths
    )

    return dataset_descriptor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pth_path",
        type=Path,
        help="path to .pth file",
    )
    parser.add_argument(
        "dataset_descriptor_file",
        type=str,
        help="path to yml dataset descriptor file",
    )
    args = parser.parse_args()

    y, cfg = YOGO.from_pth(args.pth_path)

    dataloader = load_description_to_dataloader(
        args.dataset_descriptor_file, y.Sx, y.Sy, cfg["normalize_images"]
    )

    device = choose_device()

    # These are just some standard
    config = {
        "class_names": range(y.num_classes),
        "no_classify": False,
        "healthy_weight": df.HEALTHY_WEIGHT,
        'iou_weight': df.IOU_WEIGHT,
        'no_obj_weight': df.NO_OBJ_WEIGHT,
        'classify_weight': df.CLASSIFY_WEIGHT,
        'label_smoothing': df.LABEL_SMOOTHING
    }

    with Timer("testing"):
        (
                mAP,
                confusion_data,
                accuracy,
                roc_curves,
                precision,
                recall,
                calibration_error,
        ) = Trainer._test(dataloader, device, config, args.pth_path)
