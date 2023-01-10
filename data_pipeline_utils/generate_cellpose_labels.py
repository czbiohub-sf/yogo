#! /usr/bin/env python3

import sys
import torch
import torchvision

from pathlib import Path
from cellpose import models
from cellpose.utils import (
    fill_holes_and_remove_small_masks,
    masks_to_outlines,
    outlines_list,
)


def label_folder(path_to_folder: Path):
    model = models.Cellpose(gpu=False, model_type="cyto2", device=torch.device("cpu"))
    for img_path in path_to_folder.glob("*.png"):
        print(img_path)
        img = torchvision.io.read_image(
            str(img_path), mode=torchvision.io.ImageReadMode.GRAY
        ).numpy()

        masks, _flows, _styles, _diams = model.eval([img], channels=[0,0])
        print(len(masks[0]))
        for i in range(len(masks)):
            masks[i] = fill_holes_and_remove_small_masks(masks[i])
            masks[i] = masks_to_outlines(masks[i])
            masks[i] = outlines_list(masks[i])
        return masks


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path to folder of images to label>")
        sys.exit(1)

    path_to_images = Path(sys.argv[1])
    if not path_to_images.exists():
        raise ValueError(f"{sys.argv[1]} doesn't exist")

    print(label_folder(path_to_images).pop().pop().shape)
