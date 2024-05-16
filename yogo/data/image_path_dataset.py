#! /usr/bin/env python3

import zarr
import math
import torch

import numpy as np

from torch import nn
from pathlib import Path
from collections.abc import Sized
from typing import List, Union, Optional, Callable, Tuple, cast

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from yogo.data.utils import read_image


class ImageAndIdDataset(Dataset, Sized):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        raise NotImplementedError


class ImagePathDataset(ImageAndIdDataset):
    """
    Dataset for loading images from a directory. The "__getitem__" method
    returns a tuple of (image, image_path).
    """

    def __init__(
        self,
        root: Union[str, Path],
        image_transforms: List[nn.Module] = [],
        loader: Callable[
            [
                Union[str, Path],
            ],
            torch.Tensor,
        ] = read_image,
        normalize_images: bool = False,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"{self.root} does not exist")

        self.image_paths = self.make_dataset(self.root)

        self.transform = Compose(image_transforms)
        self.loader = loader
        self.normalize_images = normalize_images

    def make_dataset(self, path_to_data: Path) -> np.ndarray:
        if path_to_data.is_file() and path_to_data.suffix == ".png":
            img_paths = [path_to_data]
        else:
            img_paths = sorted(
                [p for p in path_to_data.glob("*.png") if not p.name.startswith(".")]
            )
        if len(img_paths) == 0:
            raise FileNotFoundError(f"{str(path_to_data)} does not contain any images")
        return np.array(img_paths).astype(np.unicode_)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.image_paths[idx]
        image = self.loader(image_path)
        image = self.transform(image)
        if self.normalize_images:
            image = image / 255
        return image, image_path


class ZarrDataset(ImageAndIdDataset):
    def __init__(
        self,
        zarr_path: Union[str, Path],
        image_name_from_idx: Optional[Callable[[int], str]] = None,
        image_transforms: List[nn.Module] = [],
        normalize_images: bool = False,
    ):
        """
        Dataset for loading images from a zarr array. The "__getitem__" method
        returns a tuple of (image, image_path), where image_path is created from
        the _image_name_from_idx method.

        Note: zip files can be corrupted easily, so be aware that this
        may run into some zarr corruption issues. ImagePathDataset
        is pretty failsafe
        """
        self.zarr_path = Path(zarr_path)
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"{self.zarr_path} does not exist")

        self.zarr_store = zarr.open(str(self.zarr_path), mode="r")

        self.image_name_from_idx = image_name_from_idx or self._image_name_from_idx

        self.transform = Compose(image_transforms)
        self.normalize_images = normalize_images
        self._N = int(math.log(len(self), 10) + 1)

    def _image_name_from_idx(self, idx: int) -> str:
        return f"img_{idx:0{self._N}}.png"

    def __len__(self) -> int:
        return (
            self.zarr_store.initialized
            if isinstance(self.zarr_store, zarr.Array)
            else len(self.zarr_store)
        )

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        image = (
            self.zarr_store[:, :, idx]
            if isinstance(self.zarr_store, zarr.Array)
            else self.zarr_store[idx][:]
        )[None, ...]
        image = torch.from_numpy(image)
        image = self.transform(image)
        if self.normalize_images:
            image = image / 255

        return image, self.image_name_from_idx(idx)


def collate_fn(
    batch: List[Tuple[torch.Tensor, str]]
) -> Tuple[torch.Tensor, Tuple[str]]:
    images, fnames = zip(*batch)
    return torch.stack(images), cast(Tuple[str], fnames)


def get_dataset(
    path_to_images: Optional[Path] = None,
    path_to_zarr: Optional[Path] = None,
    image_transforms: List[nn.Module] = [],
    normalize_images: bool = False,
) -> ImageAndIdDataset:
    if path_to_images is not None and path_to_zarr is not None:
        raise ValueError(
            "can only take one of 'path_to_images' or 'path_to_zarr', but got both"
        )
    elif path_to_images is not None:
        return ImagePathDataset(
            path_to_images,
            image_transforms=image_transforms,
            normalize_images=normalize_images,
        )
    elif path_to_zarr is not None:
        return ZarrDataset(
            path_to_zarr,
            image_transforms=image_transforms,
            normalize_images=normalize_images,
        )
    else:
        raise ValueError("one of 'path_to_images' or 'path_to_zarr' must not be None")
