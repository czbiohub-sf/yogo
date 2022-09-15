import os
import csv
import yaml
import torch

from pathlib import Path
from functools import partial
from operator import itemgetter

import torchvision.transforms.functional as F

from torch import nn

from torchvision import datasets
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize
from torch.utils.data import ConcatDataset, DataLoader, random_split, Subset

from typing import Any, List, Dict, Union, Tuple, Optional, Callable, cast

from data_transforms import (
    RandomHorizontalFlipWithBBs,
    RandomVerticalFlipWithBBs,
    ImageTransformLabelIdentity,
    MultiArgSequential,
)


def load_dataset_description(
    dataset_description,
) -> Tuple[List[str], List[Dict[str, Path]], Dict[str, float]]:
    with open(dataset_description, "r") as desc:
        yaml_data = yaml.safe_load(desc)

        classes = yaml_data["class_names"]

        # either we have image_path and label_path directly defined
        # in our yaml file (describing 1 dataset exactly), or we have
        # a nested dict structure describing each dataset description.
        # see README.md for more detail
        if "dataset_paths" in yaml_data:
            dataset_paths = [
                {k: Path(v) for k, v in d.items()}
                for d in yaml_data["dataset_paths"].values()
            ]
        else:
            dataset_paths = [
                {
                    "image_path": Path(yaml_data["image_path"]),
                    "label_path": Path(yaml_data["label_path"]),
                }
            ]

        split_fractions = {
            k: float(v) for k, v in yaml_data["dataset_split_fractions"].items()
        }

        if not sum(split_fractions.values()) == 1:
            raise ValueError(
                f"invalid split fractions for dataset: split fractions must add to 1, got {split_fractions}"
            )

        check_dataset_paths(dataset_paths)
        return classes, dataset_paths, split_fractions


def check_dataset_paths(dataset_paths: List[Dict[str, Path]]):
    for dataset_desc in dataset_paths:
        if not (
            dataset_desc["image_path"].is_dir() and dataset_desc["label_path"].is_dir()
        ):
            raise FileNotFoundError(
                f"image_path or label_path do not lead to a directory\n"
                f"image_path={dataset_desc['image_path']}\nlabel_path={dataset_desc['label_path']}"
            )


def read_grayscale(img):
    return read_image(img, ImageReadMode.GRAY)


class ObjectDetectionDataset(datasets.VisionDataset):
    def __init__(
        self,
        classes: List[str],
        image_path: Path,
        label_path: Path,
        img_size: Tuple[int, int],
        loader: Callable = read_grayscale,
        extensions: Optional[Tuple[str]] = ("png",),
        is_valid_file: Optional[Callable[[str], bool]] = None,
        *args,
        **kwargs,
    ):
        # the super().__init__ just sets transforms
        # the image_path is just for repr
        super().__init__(str(image_path), *args, **kwargs)

        self.classes = classes
        self.img_size = img_size
        self.image_folder_path = image_path
        self.label_folder_path = label_path
        self.loader = loader

        self.samples = self.make_dataset(
            is_valid_file=is_valid_file, extensions=extensions
        )

    def make_dataset(
        self,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, List[List[float]]]]:
        """
        torchvision.datasets.folder.make_dataset doc string states:
            "Generates a list of samples of a form (path_to_sample, class)"

        This is designed for a dataset for classficiation (that is, mapping
        image to class), where we have a dataset for object detection (image
        to list of bounding boxes).

        Copied Pytorch's implementation of input handling[0], with changes on how we
        collect labels and images

        [0] https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html
        """
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError(
                "Both extensions and is_valid_file cannot be None or not None at the same time"
            )

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return datasets.folder.has_file_allowed_extension(x, extensions)

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        # maps file name to a list of tuples of bounding boxes + classes
        samples: List[Tuple[str, List[List[float]]]] = []
        for img_file_path in self.image_folder_path.glob("*"):
            if is_valid_file(str(img_file_path)):
                labels = self.load_labels_from_image_name(img_file_path)
                samples.append((str(img_file_path), labels))
        return samples

    def load_labels_from_image_name(self, image_path: Path) -> List[List[float]]:
        "loads labels from label file, given by image path"
        labels = []
        label_filename = image_path.name.replace(image_path.suffix, ".csv")

        # just ignore images without labels - is a missing label file
        # the best way to do this?
        try:
            with open(self.label_folder_path / label_filename, "r") as f:
                # yuck! checking for headers is not super easy
                reader = csv.reader(f)
                has_header = csv.Sniffer().has_header(f.read(1024))
                f.seek(0)
                if has_header:
                    next(reader, None)

                for row in reader:
                    assert (
                        len(row) == 5
                    ), "should have [class,xc,yc,w,h] - got length {len(row)}"
                    label_idx = self.classes.index(row[0])
                    labels.append([float(label_idx)] + [float(v) for v in row[1:]])
        except FileNotFoundError:
            pass

        return labels

    def count_class(self, class_index: int) -> int:
        s = 0
        for k, v in self.samples:
            s += sum(l[0] == class_index for l in v)
        return s

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """From torchvision.datasets.folder.DatasetFolder
        Modified (gently) to transform label as well as target
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        return sample, target

    def __len__(self) -> int:
        "From torchvision.datasets.folder.DatasetFolder"
        return len(self.samples)


def get_datasets(
    dataset_description_file: str,
    batch_size: int,
    training: bool = True,
    img_size: Tuple[int, int] = (300, 400),
    split_fractions_override: Optional[Dict[str, float]] = None,
) -> Dict[str, Subset[ConcatDataset[ObjectDetectionDataset]]]:
    (
        classes,
        dataset_paths,
        split_fractions,
    ) = load_dataset_description(dataset_description_file)

    full_dataset: ConcatDataset[ObjectDetectionDataset] = ConcatDataset(
        ObjectDetectionDataset(
            classes,
            dataset_desc["image_path"],
            dataset_desc["label_path"],
            img_size=img_size,
        )
        for dataset_desc in dataset_paths
    )

    if split_fractions_override is not None:
        split_fractions = split_fractions_override

    dataset_sizes = {
        designation: int(split_fractions[designation] * len(full_dataset))
        for designation in ["train", "val"]
    }
    test_dataset_size = {"test": len(full_dataset) - sum(dataset_sizes.values())}
    split_sizes = {**dataset_sizes, **test_dataset_size}

    assert sum(split_sizes.values()) == len(
        full_dataset
    ), f"could not create valid dataset split sizes: {split_sizes}, full dataset size is {len(full_dataset)}"

    # YUCK! Want a map from the dataset designation to teh set itself, but "random_split" takes a list
    # of lengths of dataset. So we do this verbose rigamarol.
    return dict(
        zip(
            ["train", "val", "test"],
            random_split(
                full_dataset,
                [split_sizes["train"], split_sizes["val"], split_sizes["test"]],
                generator=torch.Generator().manual_seed(42),
            ),
        )
    )


def collate_batch(batch, device="cpu", transforms=None):
    # perform image transforms here so we can transform in batches! :)
    inputs, labels = zip(*batch)
    batched_inputs = torch.stack(inputs)
    return transforms(
        batched_inputs.to(device), [torch.tensor(l).to(device) for l in labels]
    )


def get_dataloader(
    root_dir: str,
    batch_size: int,
    training: bool = True,
    img_size: Tuple[int, int] = (300, 400),
    device: Union[str, torch.device] = "cpu",
    split_fractions_override: Optional[Dict[str, float]] = None,
):
    split_datasets = get_datasets(
        root_dir,
        batch_size,
        img_size=img_size,
        training=training,
        split_fractions_override=split_fractions_override,
    )
    augmentations = (
        [RandomHorizontalFlipWithBBs(0.5), RandomVerticalFlipWithBBs(0.5)]
        if training
        else []
    )
    transforms = MultiArgSequential(
        ImageTransformLabelIdentity(Resize(img_size)), *augmentations
    )
    return {
        designation: DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=partial(collate_batch, device=device, transforms=transforms),
            shuffle=True,
            drop_last=True,
        )
        for designation, dataset in split_datasets.items()
    }
