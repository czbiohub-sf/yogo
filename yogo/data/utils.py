import torch
import warnings

from pathlib import Path
from typing import Union, Optional, Tuple, List

from torchvision.io import read_image, ImageReadMode

from yogo.data.data_transforms import (
    MultiArgSequential,
)


def read_grayscale(img_path: Union[str, Path]) -> torch.Tensor:
    try:
        return read_image(str(img_path), ImageReadMode.GRAY)
    except RuntimeError as e:
        raise RuntimeError(f"file {img_path} threw: {e}") from e


def read_grayscale_robust(
    img_path: Union[str, Path], retries: int = 3, min_duration: float = 0.1
) -> Optional[torch.Tensor]:
    """
    Attempts to read an image file in grayscale mode with retry logic.

    This function tries to read an image in grayscale mode with a specified number of retries. If all attempts fail,
    it logs a warning and returns None.
    """
    for i in range(retries):
        try:
            return read_image(str(img_path), ImageReadMode.GRAY)
        except RuntimeError as e:
            if i == retries - 1:
                warnings.warn(f"file {img_path} threw: {e}")
    return None


def collate_batch_robust(
    batch: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
    transforms: MultiArgSequential = MultiArgSequential(),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for batching data, handling optional items robustly.

    Filters out None items from a batch and applies transformations to the batched inputs and labels. This function
    is designed to work with datasets that may return None for some items, allowing the DataLoader to skip these items
    gracefully.
    """
    inputs, labels = zip(*[pair for pair in batch if pair is not None])
    batched_inputs = torch.stack(inputs)
    batched_labels = torch.stack(labels)
    return transforms(batched_inputs, batched_labels)
