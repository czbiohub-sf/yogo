#! /usr/bin/env python3

import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Tuple, List, Optional, Dict, Mapping

from torch.utils.data import Dataset

from torchvision import transforms as T
from torchvision.ops import box_iou

from yogo.data.utils import read_image_robust
from yogo.data.yogo_dataset import format_labels_tensor


PathLike = Union[str, Path]


class BlobDataset(Dataset):
    """Generates fake images from "thumbnails", which are crops of objects to be detected."""

    def __init__(
        self,
        thumbnail_dir_paths: Mapping[Union[str, int], List[PathLike]],
        Sx: int,
        Sy: int,
        classes: List[str],
        n: int = 50,
        length: int = 1000,
        background_img_shape: Tuple[int, int] = (772, 1032),
        normalize_images: bool = False,
    ):
        """
        thumbnail_dir_paths: a mapping from class (whether by class name or by class idx)
                             to a list of directories containing thumbnails for that class
        Sx, Sy: grid dimensions
        classes: list of classes
        n: number of thumbnails per class
        length: "length" of dataset
        background_img_shape: shape of background image
        normalize_images: whether to normalize images into [0,1]
        """
        super().__init__()

        self.thumbnail_dir_paths: Dict[int, List[Path]] = {
            self._convert_label(k, classes): [Path(vv) for vv in v]
            for k, v in thumbnail_dir_paths.items()
        }

        for thumbnail_dir_list in self.thumbnail_dir_paths.values():
            for thumbnail_dir in thumbnail_dir_list:
                if not Path(thumbnail_dir).exists():
                    raise FileNotFoundError(f"{str(thumbnail_dir)} does not exist")

        self.Sx = Sx
        self.Sy = Sy
        self.n = n
        self.length = length
        self.loader = read_image_robust
        self.background_img_shape = background_img_shape
        self.normalize_images = normalize_images
        self.area_threshold: int = 500

        self.classes, thumbnail_paths = self.get_thumbnail_paths(
            self.thumbnail_dir_paths
        )

        if len(self.thumbnail_dir_paths) == 0:
            raise FileNotFoundError(
                f"no thumbnails found in any of {(str(tdp) for tdp in self.thumbnail_dir_paths)}"
            )

        self.thumbnail_tensor, self.thumbnail_dims = self.load_thumbnails(
            thumbnail_paths
        )
        self.num_thumbnails = len(self.thumbnail_tensor)

    def load_thumbnails(
        self, thumbnail_paths: Tuple[Path, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a tensor of thumbnails and their dimensions"""
        with ThreadPoolExecutor() as e:
            thumbnail_list = list(
                filter(
                    lambda x: x is not None
                    and x.shape[1] * x.shape[2] > self.area_threshold,
                    tqdm(
                        e.map(self.loader, thumbnail_paths),
                        total=len(thumbnail_paths),
                        desc="loading thumbnails",
                    ),
                )
            )

        thumbnail_dims = torch.tensor(
            [thumbnail.squeeze().shape for thumbnail in thumbnail_list], dtype=torch.int
        )

        max_h, max_w = thumbnail_dims.max(0)[0]

        full_thumbnails = torch.zeros(
            (len(thumbnail_list), 1, max_h, max_w), dtype=torch.uint8
        )

        for i, (thumbnail, (h, w)) in enumerate(zip(thumbnail_list, thumbnail_dims)):
            full_thumbnails[i, :, :h, :w] = thumbnail

        return full_thumbnails, thumbnail_dims

    def _convert_label(self, label: Union[str, int], classes: List[str]) -> int:
        if isinstance(label, int):
            if not (0 <= label < len(classes)):
                raise ValueError(f"label {label} is out of range [0, {len(classes)})")
            return label

        try:
            return classes.index(label)
        except IndexError as e:
            raise ValueError(f"label {label} is not a valid YOGO class") from e

    def __len__(self) -> int:
        return self.length

    def get_thumbnail_paths(
        self, dir_paths: Dict[int, List[Path]]
    ) -> Tuple[np.ndarray, Tuple[Path, ...]]:
        "traverses down the directories and returns a list of paths to each thumbnail"
        cls_path_pairs: List[Tuple[int, Path]] = []
        for cls, list_of_data_dir in dir_paths.items():
            for data_dir in list_of_data_dir:
                if not data_dir.exists():
                    raise FileNotFoundError(f"{str(data_dir)} does not exist")

                is_empty = not any(data_dir.glob("*.png"))
                if not is_empty:
                    cls_path_pairs.extend(
                        [
                            (cls, fp)
                            for fp in data_dir.glob("*.png")
                            if not fp.name.startswith(".")
                        ]
                    )

        classes, paths = zip(*cls_path_pairs)
        return np.array(classes), paths

    def get_random_thumbnails(self, n: int = 1) -> List[Tuple[int, torch.Tensor]]:
        "Returns a list of random thumbnails and their labels"
        choices = np.random.randint(0, self.num_thumbnails, size=n)
        imgs = [
            self.thumbnail_tensor[
                i, :, : self.thumbnail_dims[i, 0], : self.thumbnail_dims[i, 1]
            ]
            for i in choices
        ]
        class_thumbnail_pairs = zip(self.classes[choices], imgs)

        return [
            (class_, img)
            for (class_, img) in class_thumbnail_pairs
            if (img.shape[0] * img.shape[1] * img.shape[2]) > 500
        ]

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

        xforms = torch.nn.Sequential(
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        )

        coords = []
        classes = []
        for class_, thumbnail in class_thumbnail_pairs:
            cp = thumbnail.clone()

            thumbnail = xforms(thumbnail).squeeze()

            if thumbnail.ndim == 1:
                raise ValueError(
                    f"thumbnail must have at least 2 dimensions - thumbnail shape is {thumbnail.shape} was {cp.shape}"
                )

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
