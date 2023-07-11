#! /usr/bin/env python3


import math
import torch
import numpy as np

from pathlib import Path
from typing import Union, Callable, Tuple, List, Optional, Dict, Mapping

from torch.utils.data import Dataset

from torchvision import transforms as T
from torchvision.ops import box_iou
from torchvision.transforms import functional as F

from yogo.data import YOGO_CLASS_ORDERING
from yogo.data.yogo_dataset import read_grayscale, format_labels_tensor


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

    def forward(self, img: torch.Tensor):
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


PathLike = Union[str, Path]


class BlobDataset(Dataset):
    def __init__(
        self,
        thumbnail_dir_paths: Mapping[Union[str, int], PathLike],
        Sx: int,
        Sy: int,
        n: int = 4,
        length: int = 1000,
        background_img_shape: Tuple[int, int] = (772, 1032),
        loader: Callable[[str], torch.Tensor] = read_grayscale,
        blend_thumbnails: bool = False,
        thumbnail_sigma: float = 1.0,
        normalize_images: bool = False,
    ):
        super().__init__()

        self.thumbnail_dir_paths: Dict[int, Path] = {
            self._convert_label(k): Path(v) for k, v in thumbnail_dir_paths.items()
        }

        for thp in self.thumbnail_dir_paths.values():
            if not thp.exists():
                raise FileNotFoundError(f"{str(thp)} does not exist")

        self.Sx = Sx
        self.Sy = Sy
        self.n = n
        self.loader = loader
        self.length = length
        self.blend_thumbnails = blend_thumbnails
        self.thumbnail_sigma = thumbnail_sigma
        self.background_img_shape = background_img_shape
        self.normalize_images = normalize_images
        self.classes, self.thumbnail_paths = self.get_thumbnail_paths(
            self.thumbnail_dir_paths
        )

        if len(self.thumbnail_dir_paths) == 0:
            raise FileNotFoundError(
                f"no thumbnails found in any of {(str(tdp) for tdp in self.thumbnail_dir_paths)}"
            )

    def _convert_label(self, label: Union[str, int]) -> int:
        if isinstance(label, int):
            if not (0 <= label < len(YOGO_CLASS_ORDERING)):
                raise ValueError(
                    f"label {label} is out of range [0, {len(YOGO_CLASS_ORDERING)})"
                )
            return label

        try:
            return YOGO_CLASS_ORDERING.index(label)
        except IndexError as e:
            raise ValueError(f"label {label} is not a valid YOGO class") from e

    def __len__(self) -> int:
        return self.length

    def get_thumbnail_paths(
        self, dir_paths: Dict[int, Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        cls_path_pairs = [
            (cls, fp) for cls, dp in dir_paths.items() for fp in dp.rglob("*.png")
        ]
        classes, paths = zip(*cls_path_pairs)
        return np.array(classes), np.array(paths).astype(np.string_)

    def get_random_thumbnails(self, n: int = 1) -> List[Tuple[int, torch.Tensor]]:
        choices = np.random.randint(0, len(self.thumbnail_paths), size=n)
        class_thumbnail_pairs = [
            (class_, self.loader(str(fp_encoded, encoding="utf-8")))
            for class_, fp_encoded in zip(
                self.classes[choices], self.thumbnail_paths[choices]
            )
        ]
        return class_thumbnail_pairs

    def get_background_shade(
        self, thumbnail: torch.Tensor, brightness_threshold: int = 210
    ) -> int:
        "rough heuristic for getting background color of thumbnail"
        val = (
            thumbnail[thumbnail > brightness_threshold]
            .float()
            .mean()
            .nan_to_num(brightness_threshold)
            .item()
        )
        return int(val)

    def propose_non_intersecting_coords(
        self,
        h: int,
        w: int,
        previous_coordinates: List[torch.Tensor],
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

        class_thumbnail_pairs = self.get_random_thumbnails(self.n)
        thumbnails = [thumbnail for _, thumbnail in class_thumbnail_pairs]

        mean_background = np.mean(
            [self.get_background_shade(thumbnail) for thumbnail in thumbnails]
        )

        img = (
            torch.empty(self.background_img_shape)
            .fill_(mean_background)
            .to(torch.uint8)
        )

        max_size = min(
            self.background_img_shape[0] // 4,
            self.background_img_shape[1] // 4,
        )
        max_scale = max_size / min(min(t.shape[-2:]) for t in thumbnails)

        if self.blend_thumbnails:
            [
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
            RandomRescale((0.5, min(max_scale, 1.5))),
        )

        # xforms = torch.jit.script(xforms)
        # TODO: Why does the above cause the following?
        # UserWarning: operator() profile_node %342 : int = prim::profile_ivalue(%out_dtype.1) does not have profile information

        coords = []
        classes = []
        for class_, thumbnail in class_thumbnail_pairs:
            thumbnail = xforms(thumbnail).squeeze()

            h, w = thumbnail.shape

            proposed_coords = self.propose_non_intersecting_coords(h, w, coords)
            if proposed_coords is None:
                continue

            x, y, normalized_coords = proposed_coords

            img[y : y + h, x : x + w] = thumbnail.to(torch.uint8)

            coords.append(normalized_coords)
            classes.append(class_)

        coords = torch.cat(coords)
        classes = torch.tensor(classes).view(-1, 1)
        coords = torch.cat([classes, coords], dim=1)
        label_tensor = format_labels_tensor(coords, self.Sx, self.Sy)

        if self.normalize_images:
            img = img / 255

        return img.unsqueeze(0), label_tensor
