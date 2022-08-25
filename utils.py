#! /usr/bin/env python3

import torch
import torchvision.transforms as T

from PIL import Image, ImageDraw
from typing import Optional,Union, List


def draw_rects(img: torch.Tensor, rects: Union[torch.Tensor, List], thresh:Optional[float]=None) -> Image:
    """
    img is the torch tensor representing an image
    rects is either
        - a torch.tensor of shape (1, pred, Sy, Sx), where pred = (xc, yc, w, h, confidence, ...)
        - a list of (class, xc, yc, w, h)
    thresh is a threshold for confidence when rects is a torch.Tensor
    """
    assert (
        len(img.shape) == 2
    ), f"takes single grayscale image - should be 2d, got {img.shape}"
    h, w = img.shape

    if isinstance(rects, torch.Tensor):
        _, pred_dim, Sy, Sx = img.shape
        if thresh is None: thresh = 0.
        rects = [r for r in img[0, ...].reshape(Sy * Sx, pred_dim) if r[4] > thresh]
    elif isinstance(rects, list):
        if thresh is not None:
            raise ValueError("threshold only valid for tensor (i.e. prediction) input")
        rects = [r[1:] for r in rects]

    formatted_rects = [
        [
            int(w * (r[0] - r[2] / 2)),
            int(h * (r[1] - r[3] / 2)),
            int(w * (r[0] + r[2] / 2)),
            int(h * (r[1] + r[3] / 2)),
        ]
        for r in rects
    ]

    image = T.ToPILImage()(img[None, ...])
    rgb = Image.new("RGB", image.size)
    rgb.paste(image)
    draw = ImageDraw.Draw(rgb)

    for r in formatted_rects:
        draw.rectangle(r, outline="red")

    return rgb
