#! /usr/bin/env python3


import torch
import numpy as np

from pathlib import Path
from typing import Union, Callable, Tuple

from torch.utils.data import Dataset

from torchvision import transforms as T
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import functional as F

from yogo.data.dataset import read_grayscale, format_labels_tensor


class RandomRescale(torch.nn.Module):
    def __init__(
        self,
        scale: Tuple[int, int],
        p=0.5,
        interpolation=F.InterpolationMode.BILINEAR,
        max_size=None,
        antialias="warn",
    ):
        super().__init__()

        self.scale = scale
        self.max_size = max_size

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        if torch.rand(1) > self.p:
            return img

        img_size = torch.tensor(img.shape[-2:])
        scale = torch.rand(1) * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = (img_size * scale).int().tolist()
        return F.resize(img, size, self.interpolation, self.max_size, self.antialias)

    def __repr__(self) -> str:
        detail = f"(scale={self.scale}, interpolation={self.interpolation.value}, max_size={self.max_size}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"


class BlobDataset(Dataset):
    def __init__(
        self,
        misc_thumbnail_path: Union[str, Path],
        length: int = 1000,
        background_img_shape: tuple[int, int] = (772, 1032),
        loader: Callable[[str], torch.Tensor] = read_grayscale,
        normalize_images: bool = False,
        label: int = 6,
    ):
        super().__init__()
        self.misc_thumbnail_path = Path(misc_thumbnail_path)

        if not self.misc_thumbnail_path.exists():
            raise FileNotFoundError(f"{misc_thumbnail_path} does not exist")

        self.label = label
        self.loader = loader
        self.length = length
        self.thumbnail_paths = self.get_thumbnail_paths(self.misc_thumbnail_path)

    def __len__(self) -> int:
        return self.length

    def get_thumbnail_paths(self, folder_path: Path):
        paths = [fp for fp in folder_path.glob("*.png")]
        return np.array(paths).astype(np.string_)

    def get_random_thumbnails(self, n: int = 1) -> list[torch.Tensor]:
        paths = [
            str(fp_encoded, encoding="utf-8")
            for fp_encoded in np.random.choice(self.thumbnail_paths, size=n)
        ]
        return [self.loader(fp) for fp in paths]

    def get_background_shade(
        self, thumbnail: torch.Tensor, darkness_threshold: int = 200
    ) -> int:
        "rough heuristic for getting background color of thumbnail"
        val = thumbnail[thumbnail > darkness_threshold].mean().item()
        return int(val)

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(f"index {idx} is out of bounds for length {self.length}")

        thumbnails = self.get_random_thumbnails()
        mean_background = np.mean(
            [self.get_background_shade(thumbnail) for thumbnail in thumbnails]
        )

        img = torch.fill_(torch.empty(self.background_img_shape), mean_background)

        max_size = (
            self.background_img_shape[0] // 4,
            self.background_img_shape[1] // 4,
        )

        xforms = torch.nn.Sequential(
            T.RandomRotation(180, expand=True, fill=mean_background),
            RandomRescale((0.8, 2.0), p=0.5, max_size=max_size),
        )
        xforms = torch.jit.script(xforms)

        coords = []
        for thumbnail in thumbnails:
            thumbnail = xforms(thumbnail)
            h, w = thumbnail.shape
            x = np.random.randint(0, self.background_img_shape[0] - w)
            y = np.random.randint(0, self.background_img_shape[1] - h)
            img[x : x + w, y : y + h] = thumbnail
            coords.append([self.label, x, y, x + w, y + h])

        label_tensor = format_labels_tensor(torch.tensor(coords))

        return img, label_tensor
