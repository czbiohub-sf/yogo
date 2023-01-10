#! /usr/bin/env python3

import cv2
import sys
import torch
import torchvision

import cellpose

import numpy as np

from tqdm import tqdm
from typing import Sequence, Generator, List, TypeVar, Tuple
from pathlib import Path
from cellpose import models
from cellpose import io
from cellpose.utils import (
    fill_holes_and_remove_small_masks,
    masks_to_outlines,
    outlines_list,
)

T = TypeVar("T")


def iter_in_chunks(s: Sequence[T], n: int = 1) -> Generator[Sequence[T], None, None]:
    for i in range(0, len(s), n):
        yield s[i : i + n]


def get_outlines(path_to_folder: Path, chunksize: int = 32) -> List[Tuple[Path, List[np.ndarray]]]:
    model = models.Cellpose(gpu=True, model_type="cyto2", device=torch.device("cuda"))

    outlines: List[Tuple[Path, List[np.ndarray]]] = []

    image_filenames = list(path_to_folder.glob("*.png"))
    filename_iterator = iter_in_chunks(image_filenames, chunksize)

    for img_filename_chunk in filename_iterator:
        imgs = [
            cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            for img_path in img_filename_chunk
        ]
        masks, _flows, _styles, _diams = model.eval(imgs, channels=[0, 0])

        for file_path, mask in zip(img_filename_chunk, masks):
            refined_mask = fill_holes_and_remove_small_masks(mask)
            mask_outlines = outlines_list(refined_mask)
            outlines.append((file_path, mask_outlines))

    return outlines


def to_bb_labels(bb_csv_fd, outlines, label):
    for file_path, image_outlines in outlines:
        for outline in image_outlines:
            xmin, xmax, ymin, ymax = (
                outline[:, 0].min(),
                outline[:, 0].max(),
                outline[:, 1].min(),
                outline[:, 1].max(),
            )
            bb_csv_fd.write(f"{str(file_path)},{xmin},{xmax},{ymin},{ymax},{label},0,0\n")


def label_folder(path_to_csv, path_to_images, chunksize=32, label=0):
    outlines = get_outlines(path_to_images, chunksize=chunksize)
    with open(str(path_to_csv), "w") as f:
        f.write("image_id,xmin,xmax,ymin,ymax,label,prob,unique_cell_id\n")
        to_bb_labels(f, outlines, label)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path to folder of images to label>")
        sys.exit(1)

    path_to_images = Path(sys.argv[1])
    if not path_to_images.exists():
        raise ValueError(f"{sys.argv[1]} doesn't exist")

    path_to_csv = path_to_images.parent / "labels.csv"
    label_folder(path_to_csv, path_to_images)
