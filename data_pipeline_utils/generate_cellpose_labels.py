#! /usr/bin/env python3

import cv2
import sys
import time
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
    remove_edge_masks,
    outlines_list,
)

from labelling_constants import CLASSES
from utils import normalize, convert_coords, multiprocess_directory_work

T = TypeVar("T")


def iter_in_chunks(s: Sequence[T], n: int = 1) -> Generator[Sequence[T], None, None]:
    for i in range(0, len(s), n):
        yield s[i : i + n]


def get_outlines(
    path_to_folder: Path, chunksize: int = 32
) -> List[Tuple[Path, List[np.ndarray]]]:
    """ Return a list of tuples (path to image, detection outlines)

    This should be run on the GPU, else it is painfully slow! Allocate some CPU too,
    we are doing a good amount of image processing.
    """
    model = models.Cellpose(gpu=True, model_type="cyto2", device=torch.device("cuda"))

    outlines: List[Tuple[Path, List[np.ndarray]]] = []

    image_filenames = list(path_to_folder.glob("*.png"))
    filename_iterator = iter_in_chunks(image_filenames, chunksize)

    for img_filename_chunk in filename_iterator:
        imgs = [
            cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            for img_path in img_filename_chunk
        ]

        # flows, styles, and diameters are not used
        per_img_masks, _flows, _styles, _diams = model.eval(imgs, channels=[0, 0])

        for file_path, masks in zip(img_filename_chunk, per_img_masks):
            masks = fill_holes_and_remove_small_masks(masks)
            masks = remove_edge_masks(masks)
            mask_outlines = outlines_list(masks)
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
            bb_csv_fd.write(
                f"{str(file_path)},{xmin},{xmax},{ymin},{ymax},{label},0,0\n"
            )


def to_yogo_labels(label_dir_path, outlines, label):
    for file_path, image_outlines in outlines:
        label_file_name = str(label_dir_path / file_path.with_suffix(".txt").name)
        with open(label_file_name, "w") as f:
            for outline in image_outlines:
                xmin, xmax, ymin, ymax = (
                    outline[:, 0].min(),
                    outline[:, 0].max(),
                    outline[:, 1].min(),
                    outline[:, 1].max(),
                )
                xcenter, ycenter, width, height = convert_coords(xmin, xmax, ymin, ymax)
                f.write(f"{label} {xcenter} {ycenter} {width} {height}\n")


def label_folder_for_yogo(path_to_images: Path, chunksize=32, label=0):
    # Write classes.txt for label studio
    with open(str(path_to_images.parent / "classes.txt"), "w") as f:
        for clss in CLASSES:
            f.write(f"{clss}\n")

    path_to_label_dir = path_to_images.parent / "labels"
    path_to_label_dir.mkdir(exist_ok=False, parents=False)

    outlines = get_outlines(path_to_images, chunksize=chunksize)

    to_yogo_labels(path_to_label_dir, outlines, label)


def label_folder_for_napari(path_to_images: Path, chunksize=32, label=0):
    "most likely depricated!"
    outlines = get_outlines(path_to_images, chunksize=chunksize)

    path_to_csv = path_to_images.parent / "labels.csv"
    with open(str(path_to_csv), "w") as f:
        f.write("image_id,xmin,xmax,ymin,ymax,label,prob,unique_cell_id\n")
        to_bb_labels(f, outlines, label)


def label_runset(path_to_runset_folder: Path, chunksize=32, label=0):
    print("finding directories to label...")
    files = list(path_to_runset_folder.glob(f"./**/images"))
    print(f"found {len(files)} directories to label")

    for i, f in enumerate(files, start=1):
        print(f"{i} / {len(files)} | {f.parent.name}", end="    ")
        t0 = time.perf_counter()

        label_dir = f.parent / "labels"
        if label_dir.exists():
            print(f"skipping {f} since label directory {label_dir} exists")
            continue

        try:
            label_folder_for_yogo(f, chunksize=chunksize, label=label)
        except Exception as e:
            print(e)
        print(f"{time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path to runset>")
        sys.exit(1)

    path_to_images = Path(sys.argv[1])

    if not path_to_images.exists():
        raise ValueError(f"{str(path_to_images)} doesn't exist")

    label_runset(path_to_images)
