import torch

from pathlib import Path


from torchvision.io import read_image, ImageReadMode

from typing import Union


def read_grayscale(img_path: Union[str, Path]) -> torch.Tensor:
    try:
        return read_image(str(img_path), ImageReadMode.GRAY)
    except RuntimeError as e:
        raise RuntimeError(f"file {img_path} threw: {e}")
