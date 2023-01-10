#! /usr/bin/env python3

import cv2
import sys
import torch
import torchvision

import cellpose

from tqdm import tqdm
from typing import Sequence, Generator, List, TypeVar
from pathlib import Path
from cellpose import models
from cellpose import io
from cellpose.utils import (
    fill_holes_and_remove_small_masks,
    masks_to_outlines,
    outlines_list,
)

T = TypeVar("T")


def iter_in_chunks(s: Sequence[T], n: int = 1) -> Generator[List[T], None, None]:
    for i in range(0, len(s), n):
        yield s[i : i + n]


def label_folder(path_to_folder: Path, chunksize: int = 32):
    model = models.Cellpose(gpu=True, model_type="cyto2", device=torch.device("cuda"))

    outlines = []

    image_filenames = list(path_to_folder.glob("*.png"))
    filename_iterator = iter_in_chunks(image_filenames, chunksize)
    filename_iterator_num_chunks = len(image_filenames) // chunksize + 1

    for img_filename_chunk in tqdm(
        filename_iterator, total=filename_iterator_num_chunks
    ):
        imgs = [
            cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            for img_path in img_filename_chunk
        ]
        masks, _flows, _styles, _diams = model.eval(imgs, channels=[0, 0])

        for mask in masks:
            refined_mask = fill_holes_and_remove_small_masks(mask)
            mask_outlines = outlines_list(refined_mask)
            outlines.append(mask_outlines)

    return outlines


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path to folder of images to label>")
        sys.exit(1)

    path_to_images = Path(sys.argv[1])
    if not path_to_images.exists():
        raise ValueError(f"{sys.argv[1]} doesn't exist")

    outlines = label_folder(path_to_images)
    print(len(outlines))
