#! /usr/bin/env python3


import math
import torch
import numpy as np

from pathlib import Path
from typing import Union, Callable, Tuple, List, Optional

from torch.utils.data import Dataset

from torchvision import transforms as T
from torchvision.ops import box_iou
from torchvision.transforms import functional as F

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
        scale = (torch.rand(1) * (self.scale[1] - self.scale[0]) + self.scale[0]).item()
        new_img_shape = [int(v) for v in img_size * scale]
        return F.resize(
            img,
            size=new_img_shape,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )

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
        background_img_shape: Tuple[int, int] = (772, 1032),
        loader: Callable[[str], torch.Tensor] = read_grayscale,
        blend_thumbnails: bool = False,
        thumbnail_sigma: float = 1.0,
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
        self.blend_thumbnails = blend_thumbnails
        self.thumbnail_sigma = thumbnail_sigma
        self.background_img_shape = background_img_shape
        self.thumbnail_paths = self.get_thumbnail_paths(self.misc_thumbnail_path)

        if len(self.thumbnail_paths) == 0:
            raise FileNotFoundError(f"no thumbnails found in {misc_thumbnail_path}")

    def __len__(self) -> int:
        return self.length

    def get_thumbnail_paths(self, folder_path: Path):
        paths = [fp for fp in folder_path.glob("*.png")]
        return np.array(paths).astype(np.string_)

    def get_random_thumbnails(self, n: int = 1) -> List[torch.Tensor]:
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
        h: int,
        w: int,
        previous_coordinates: List[List[float]],
        num_tries: int = 100,
    ) -> Optional[Tuple[int, int, torch.Tensor]]:
        while num_tries > 0:
            y = np.random.randint(0, self.background_img_shape[0] - h)
            x = np.random.randint(0, self.background_img_shape[1] - w)
            normalized_coords = torch.tensor(
                [
                    [
                        x / self.background_img_shape[1],
                        y / self.background_img_shape[0],
                        (x + w) / self.background_img_shape[1],
                        (y + h) / self.background_img_shape[0],
                    ]
                ]
            )
            if len(previous_coordinates) == 0 or box_iou(
                normalized_coords, torch.cat(previous_coordinates)
            ).sum().eq(0):
                return x, y, normalized_coords
            num_tries -= 1
        return None

    def gaussian_kernel(self, thumbnail, thumbnail_sigma):
        def gaussian(x, mu=0.0, sigma=1.0):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (
                sigma * math.sqrt(2 * math.pi)
            )

        w = torch.linspace(-1, 1, thumbnail.shape[1])
        h = torch.linspace(-1, 1, thumbnail.shape[0])

        w_gaussian = gaussian(w, sigma=thumbnail_sigma)
        h_gaussian = gaussian(h, sigma=thumbnail_sigma)

        kernel = torch.sqrt(h_gaussian[:, None] * w_gaussian[None, :])
        kernel = (kernel - kernel.min()) * (1 / (kernel.max() - kernel.min()))

        return kernel

    def square_kernel(self, thumbnail):
        "not as good as gaussian kernel"
        hs = torch.linspace(-1, 1, thumbnail.shape[0])
        ws = torch.linspace(-1, 1, thumbnail.shape[1])
        kernel = torch.sqrt(hs[:, None] ** 2 * ws[None, :] ** 2)
        kernel = torch.where(kernel < 0.5, 0, kernel)
        kernel = (kernel - kernel.min()) * (1 / (kernel.max() - kernel.min()))
        kernel = 1 - kernel

        return kernel

    def blend_thumbnail(
        self, bg_shade: int, thumbnail: torch.Tensor, thumbnail_sigma: float = 1.0
    ) -> torch.Tensor:
        """Michelle's Idea"""
        kernel = self.gaussian_kernel(thumbnail, thumbnail_sigma)
        return bg_shade * (1 - kernel) + thumbnail * kernel

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

        if self.blend_thumbnails:
            thumbnails = [
                self.blend_thumbnail(
                    mean_background,
                    thumbnail.squeeze(),
                    thumbnail_sigma=self.thumbnail_sigma,
                ).unsqueeze(0)
                for thumbnail in thumbnails
            ]

        xforms = torch.nn.Sequential(
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            # TODO this mucks w/ bounding boxes for rectangular shapes, probably ok for rbcs
            # T.RandomRotation(180, expand=True, fill=mean_background),
            RandomRescale((0.8, min(max_scale, 2))),
        )

        # xforms = torch.jit.script(xforms)
        # TODO: Why does the above cause the following?
        # UserWarning: operator() profile_node %342 : int = prim::profile_ivalue(%out_dtype.1) does not have profile information

        coords = []
        for thumbnail in thumbnails:
            thumbnail = xforms(thumbnail).squeeze()

            h, w = thumbnail.shape

            proposed_coords = self.propose_non_intersecting_coords(h, w, coords)
            if proposed_coords is None:
                continue

            x, y, normalized_coords = proposed_coords

            img[y : y + h, x : x + w] = thumbnail

            coords.append(normalized_coords)

        coords = torch.cat(coords)
        coords = torch.cat([torch.ones(coords.shape[0], 1), coords], dim=1)
        label_tensor = format_labels_tensor(coords, self.Sx, self.Sy)

        return img.unsqueeze(0), label_tensor
