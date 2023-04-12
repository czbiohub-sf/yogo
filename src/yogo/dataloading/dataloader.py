import os
import torch

from tqdm import tqdm
from ruamel import yaml
from pathlib import Path
from functools import partial
from dataclasses import dataclass

from torchvision.transforms import Resize, RandomAdjustSharpness, ColorJitter
from torch.utils.data import ConcatDataset, DataLoader, random_split, Subset

from typing import List, Dict, Union, Tuple, Optional, Literal

from yogo.dataloading.dataset import ObjectDetectionDataset
from yogo.dataloading.data_transforms import (
    DualInputModule,
    DualInputId,
    RandomHorizontalFlipWithBBs,
    RandomVerticalFlipWithBBs,
    RandomVerticalCrop,
    ImageTransformLabelIdentity,
    MultiArgSequential,
)


DatasetSplitName = Literal["train", "val", "test"]


class InvalidDatasetDescriptionFile(Exception):
    ...


@dataclass
class DatasetDescription:
    classes: List[str]
    split_fractions: Dict[str, float]
    dataset_paths: List[Dict[str, Path]]
    test_dataset_paths: Optional[List[Dict[str, Path]]]

    def __iter__(self):
        return iter((self.classes, self.split_fractions, self.dataset_paths, self.test_dataset_paths))


def load_dataset_description(dataset_description: str) -> DatasetDescription:
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

        if "test_paths" in yaml_data:
            if "dataset_split_fractions" in yaml_data:
                raise ValueError(
                    "when test_paths have been given explicitly, ")

            test_dataset_paths = [
                {k: Path(v) for k, v in d.items()}
                for d in yaml_data["test_paths"].values()
            ]
        else:

            test_dataset_paths = None

        split_fractions = {
            k: float(v) for k, v in yaml_data["dataset_split_fractions"].items()
        }

        if not sum(split_fractions.values()) == 1:
            raise ValueError(
                "invalid split fractions for dataset: split fractions must add to 1, "
                f"got {split_fractions}"
            )

        check_dataset_paths(dataset_paths, prune=True)
        check_dataset_paths(test_dataset_paths, prune=False)

        return DatasetDescription(
            classes, split_fractions, dataset_paths, test_dataset_paths)


def check_dataset_paths(dataset_paths:List[Dict[str, Path]],prune:bool=False):
    to_prune: List[int] = []
    for i in range(len(dataset_paths)):
        if not (
            dataset_paths[i]["image_path"].is_dir()
            and dataset_paths[i]["label_path"].is_dir()
            and len(list(dataset_paths[i]["label_path"].iterdir())) > 0
        ):
            if prune:
                print(f"pruning {dataset_paths[i]}")
                to_prune.append(i)
            else:
                raise FileNotFoundError(
                    f"image_path or label_path do not lead to a directory\n"
                    f"image_path={dataset_paths['image_path']}\nlabel_path={dataset_paths['label_path']}"
                )

    # reverse order so we don't move around the to-delete items in the list
    for i in to_prune[::-1]:
        del dataset_paths[i]


def get_datasets(
    dataset_description_file: str,
    Sx: int,
    Sy: int,
    split_fractions_override: Optional[Dict[str, float]] = None,
    normalize_images: bool = False,
) -> Dict[DatasetSplitName, Subset[ConcatDataset[ObjectDetectionDataset]]]:
    dataset_desc = load_dataset_description(dataset_description_file)
    (
        dataset_classes,
        split_fractions,
        dataset_paths,
        test_dataset_paths,
    ) = dataset_desc

    # can we speed this up? multiproc dataset creation?
    full_dataset: ConcatDataset[ObjectDetectionDataset] = ConcatDataset(
        ObjectDetectionDataset(
            dataset_classes,
            dataset_paths["image_path"],
            dataset_paths["label_path"],
            Sx,
            Sy,
            normalize_images=normalize_images,
        )
        for dataset_paths in tqdm(dataset_paths)
    )

    if split_fractions_override is not None:
        split_fractions = split_fractions_override

    dataset_sizes = {
        designation: round(split_fractions[designation] * len(full_dataset))
        for designation in ["train", "val"]
    }
    test_dataset_size = {"test": len(full_dataset) - sum(dataset_sizes.values())}
    split_sizes = {**dataset_sizes, **test_dataset_size}

    assert all([sz > 0 for sz in split_sizes.values()]) and sum(
        split_sizes.values()
    ) == len(
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
                generator=torch.Generator().manual_seed(111111),
            ),
        )
    )


def collate_batch(batch, device="cpu", transforms=None):
    # TODO https://pytorch.org/docs/stable/data.html#memory-pinning
    # perform image transforms here so we can transform in batches! :)
    inputs, labels = zip(*batch)
    batched_inputs = torch.stack(inputs)
    batched_labels = torch.stack(labels)
    return transforms(batched_inputs, batched_labels)


def get_dataloader(
    dataset_descriptor_file: str,
    batch_size: int,
    Sx: int,
    Sy: int,
    training: bool = True,
    preprocess_type: Optional[str] = None,
    vertical_crop_size: Optional[float] = None,
    resize_shape: Optional[Tuple[int, int]] = None,
    device: Union[str, torch.device] = "cpu",
    split_fractions_override: Optional[Dict[str, float]] = None,
    normalize_images: bool = False,
) -> Dict[DatasetSplitName, DataLoader]:
    split_datasets = get_datasets(
        dataset_descriptor_file,
        Sx,
        Sy,
        split_fractions_override=split_fractions_override,
        normalize_images=normalize_images,
    )
    augmentations = (
        [
            ImageTransformLabelIdentity(RandomAdjustSharpness(0, p=0.5)),
            ImageTransformLabelIdentity(ColorJitter(brightness=0.2, contrast=0.2)),
            RandomHorizontalFlipWithBBs(0.5),
            RandomVerticalFlipWithBBs(0.5),
        ]
        if training
        else []
    )

    image_preprocess: DualInputModule
    if preprocess_type == "crop":
        assert vertical_crop_size is not None, "must be None if cropping"
        image_preprocess = RandomVerticalCrop(vertical_crop_size)
    elif preprocess_type == "resize":
        image_preprocess = Resize(resize_shape)
    elif preprocess_type is None:
        image_preprocess = DualInputId()
    else:
        raise ValueError(f"got invalid preprocess type {preprocess_type}")

    d = dict()
    for designation, dataset in split_datasets.items():
        transforms = MultiArgSequential(
            image_preprocess,
            *augmentations if designation == "train" else [],
        )
        d[designation] = DataLoader(
            dataset,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            batch_size=batch_size,
            persistent_workers=True,
            multiprocessing_context="spawn",
            # optimal # of workers?
            num_workers=max(4, min(len(os.sched_getaffinity(0)) // 2, 16)),  # type: ignore
            generator=torch.Generator().manual_seed(111111),
            collate_fn=partial(collate_batch, device=device, transforms=transforms),
        )
    return d
