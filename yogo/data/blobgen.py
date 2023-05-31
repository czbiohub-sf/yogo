#! /usr/bin/env python3


import torch
import numpy as np

from pathlib import Path
from typing import Union, Callable, Tuple, Optional

from torch.utils.data import Dataset

from torchvision import transforms as T
from torchvision.ops import box_iou
from torchvision.transforms import functional as F
from torchvision.io import read_image, ImageReadMode

from yogo.data.dataset import read_grayscale, format_labels_tensor


class RandomRescale(torch.nn.Module):
    def __init__(
        self,
        scale: Tuple[int, int],
        interpolation=F.InterpolationMode.BILINEAR,
        antialias=True,
    ):
        super().__init__()

        self.scale = scale

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        img_size = torch.tensor(img.shape[-2:])
        scale  = (torch.rand(1) * (self.scale[1] - self.scale[0]) + self.scale[0]).item()
        new_img_shape = [int(v) for v in img_size * scale]
        return F.resize(img, size=new_img_shape, interpolation=self.interpolation, antialias=self.antialias)

    def __repr__(self) -> str:
        detail = f"(scale={self.scale}, interpolation={self.interpolation.value}, max_size={self.max_size}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"


class BlobDataset(Dataset):
    def __init__(
        self,
        misc_thumbnail_path: Union[str, Path],
        Sx: int,
        Sy: int,
        n: int = 4,
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

        self.Sx = Sx
        self.Sy = Sy
        self.n = n
        self.label = label
        self.loader = loader
        self.length = length
        self.background_img_shape = background_img_shape
        self.thumbnail_paths = self.get_thumbnail_paths(self.misc_thumbnail_path)

        if len(self.thumbnail_paths) == 0:
            raise FileNotFoundError(f"no thumbnails found in {misc_thumbnail_path}")

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
        self, thumbnail: torch.Tensor, darkness_threshold: int = 210
    ) -> int:
        "rough heuristic for getting background color of thumbnail"
        val = thumbnail[thumbnail > darkness_threshold].float().mean().item()
        return int(val)

    def propose_non_intersecting_coords(
            self,
            previous_coordinates: torch.Tensor,
            h: int,
            w: int,
            num_tries: int = 100,
        ) -> Optional[Tuple[int, int, torch.Tensor]]:
        while num_tries > 0:
            y = np.random.randint(0, self.background_img_shape[0] - h)
            x = np.random.randint(0, self.background_img_shape[1] - w)
            normalized_coords = torch.tensor([[
                x / self.background_img_shape[1],
                y / self.background_img_shape[0],
                (x + w) / self.background_img_shape[1],
                (y + h) / self.background_img_shape[0],
            ]])
            if previous_coordinates.sum().eq(0) or box_iou(normalized_coords, previous_coordinates).sum().eq(0):
                return x, y, normalized_coords
            num_tries -= 1
        return None


    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(f"index {idx} is out of bounds for length {self.length}")

        thumbnails = self.get_random_thumbnails(self.n)
        mean_background = np.mean(
            [self.get_background_shade(thumbnail) for thumbnail in thumbnails]
        )

        img = torch.fill_(torch.empty(self.background_img_shape), mean_background)

        max_size = min(
            self.background_img_shape[0] // 4,
            self.background_img_shape[1] // 4,
        )
        max_scale = max_size / min(min(t.shape[-2:]) for t in thumbnails)

        xforms = torch.nn.Sequential(
            T.RandomRotation(180, expand=True, fill=mean_background),
            RandomRescale((0.8, min(max_scale, 2))),
        )
        xforms = torch.jit.script(xforms)

        coords = torch.empty((len(thumbnails), 5))
        for i, thumbnail in enumerate(thumbnails):
            thumbnail = xforms(thumbnail)
            _, h, w = thumbnail.shape

            proposed_coords = self.propose_non_intersecting_coords(coords[:, 1:], h, w)
            if proposed_coords is None:
                print('continuing')
                continue

            x, y, normalized_coords = proposed_coords

            img[y : y + h, x : x + w] = thumbnail[0]

            coords[i, 0] = self.label
            coords[i, 1:] = normalized_coords

        label_tensor = format_labels_tensor(coords, self.Sx, self.Sy)

        return img, label_tensor
