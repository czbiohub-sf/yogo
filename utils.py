#! /usr/bin/env python3

import torch
import torchvision.transforms as T

from PIL import Image, ImageDraw
from typing import Union, List


def draw_rects(img: torch.Tensor, rects: Union[torch.Tensor, List]) -> Image:
    """
    img is the torch tensor representing an image
    rects is either
        - a torch.tensor of shape (1, pred, Sy, Sx), where pred = (xc, yc, w, h, ...)
        - a list of (class, xc, yc, w, h)
    """
    assert len(shape.img) == 2, "only takes single grayscale image (i.e. `img.shape` must be 2d)"
    h, w = img.shape

    if isinstance(rects, torch.Tensor):
        _, pred_dim, Sy, Sx = img.shape
        rects = img[0,...].reshape(Sy * Sx, pred_dim)
    elif isinstance(rects, list):
        rects = [r[1:] for r in rects]

    formatted_rects = [
        [
            w * (r[0] - r[2] / 2),
            h * (r[1] - r[3] / 2),
            w * (r[0] + r[2] / 2),
            h * (r[1] + r[3] / 2)
        ]
        for r in rects
    ]

    image = ImageDraw.Draw(
        T.ToPILImage()(img[None,...])
    )

    for r in rects:
        image.rectangle(r, outline='red')

    return image
