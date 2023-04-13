import os
import torch

from tqdm import tqdm
from ruamel import yaml
from pathlib import Path
from functools import partial
from dataclasses import dataclass

from torchvision.transforms import Resize, RandomAdjustSharpness, ColorJitter
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split

from typing import List, Dict, Union, Tuple, Optional

from yogo.data.dataset import ObjectDetectionDataset
from yogo.data.data_transforms import (
    DualInputModule,
    DualInputId,
    RandomHorizontalFlipWithBBs,
    RandomVerticalFlipWithBBs,
    RandomVerticalCrop,
    ImageTransformLabelIdentity,
    MultiArgSequential,
)


class InvalidDatasetDescriptionFile(Exception):
    ...


@dataclass
class DatasetDescription:
    classes: List[str]
    split_fractions: Dict[str, float]
    dataset_paths: List[Dict[str, Path]]
    test_dataset_paths: Optional[List[Dict[str, Path]]]

    def __iter__(self):
        return iter(
            (
                self.classes,
                self.split_fractions,
                self.dataset_paths,
                self.test_dataset_paths,
            )
        )


def check_dataset_paths(dataset_paths: List[Dict[str, Path]], prune: bool = False):
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
                    f"image_path={dataset_paths[i]['image_path']}\nlabel_path={dataset_paths[i]['label_path']}"
                )

    # reverse order so we don't move around the to-delete items in the list
    for i in to_prune[::-1]:
        del dataset_paths[i]


def load_dataset_description(dataset_description: str) -> DatasetDescription:
    """Loads and validates dataset description file"""
    required_keys = [
        "class_names",  # we don't actually use classes fm dataset desc files, but keep for now
        "dataset_split_fractions",
        "dataset_paths",
    ]
    with open(dataset_description, "r") as desc:
        yaml_data = yaml.safe_load(desc)

        for k in required_keys:
            if k not in yaml_data:
                raise InvalidDatasetDescriptionFile(
                    f"{k} is required in dataset description files, but was "
                    f"found missing for {dataset_description}"
                )

        classes = yaml_data["class_names"]
        split_fractions = {
            k: float(v) for k, v in yaml_data["dataset_split_fractions"].items()
        }
        dataset_paths = [
            {k: Path(v) for k, v in d.items()}
            for d in yaml_data["dataset_paths"].values()
        ]
        check_dataset_paths(dataset_paths, prune=True)

        if "test_paths" in yaml_data:
            test_dataset_paths = [
                {k: Path(v) for k, v in d.items()}
                for d in yaml_data["test_paths"].values()
            ]
            check_dataset_paths(test_dataset_paths, prune=False)

            # when we have 'test_paths', all the data from dataset_paths
            # will be used for training, so we should only have 'test' and
            # 'val' in dataset_split_fractions.
            if "val" not in split_fractions or "test" not in split_fractions:
                raise InvalidDatasetDescriptionFile(
                    "'val' and 'test' are required keys for dataset_split_fractions"
                )
            if "train" in split_fractions:
                raise InvalidDatasetDescriptionFile(
                    "when `test_paths` is present in a dataset descriptor file, 'train' "
                    "is not a valid key for `dataset_split_fractions`, since we will use "
                    "all the data from `dataset_paths` for training"
                )
        else:
            test_dataset_paths = None
            if any(k not in split_fractions for k in ("test", "train", "val")):
                raise InvalidDatasetDescriptionFile(
                    "'train', 'val', and 'test' are required keys for dataset_split_fractions - missing at least one. "
                    f"split fractions was {split_fractions}"
                )

        if not sum(split_fractions.values()) == 1:
            raise InvalidDatasetDescriptionFile(
                "invalid split fractions for dataset: split fractions must add to 1, "
                f"got {split_fractions}"
            )

        return DatasetDescription(
            classes, split_fractions, dataset_paths, test_dataset_paths
        )


def get_datasets(
    dataset_description_file: str,
    Sx: int,
    Sy: int,
    split_fractions_override: Optional[Dict[str, float]] = None,
    normalize_images: bool = False,
) -> Dict[str, Dataset]:
    (
        dataset_classes,
        split_fractions,
        dataset_paths,
        test_dataset_paths,
    ) = load_dataset_description(dataset_description_file)

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

    if test_dataset_paths is not None:
        test_dataset: ConcatDataset[ObjectDetectionDataset] = ConcatDataset(
            ObjectDetectionDataset(
                dataset_classes,
                dataset_paths["image_path"],
                dataset_paths["label_path"],
                Sx,
                Sy,
                normalize_images=normalize_images,
            )
            for dataset_paths in tqdm(test_dataset_paths)
        )
        return {
            "train": full_dataset,
            **split_dataset(test_dataset, split_fractions),
        }

    return split_dataset(full_dataset, split_fractions)


def split_dataset(
    dataset: Dataset, split_fractions: Dict[str, float]
) -> Dict[str, Dataset]:
    if not hasattr(dataset, "__len__"):
        raise ValueError(
            f"dataset {dataset} must have a length (specifically, `__len__` must be defined)"
        )

    if len(split_fractions) == 0:
        raise ValueError("must have at least one value for the split!")
    elif len(split_fractions) == 1:
        if not next(iter(split_fractions)) == 1:
            raise ValueError(
                "when split_fractions has length 1, it must have a value of 1"
            )
        keys = list(split_fractions)
        return {keys.pop(): dataset}

    keys = list(split_fractions)

    # very annoying type hint here - `Dataset` doesn't necessarily have `__len__`,
    # so we manually check it. But I am not sure that you can cast to Sizedj so mypy complains
    dataset_sizes = {
        k: round(split_fractions[k] * len(dataset)) for k in keys[:-1]  # type: ignore
    }
    final_dataset_size = {keys[-1]: len(dataset) - sum(dataset_sizes.values())}  # type: ignore
    split_sizes = {**dataset_sizes, **final_dataset_size}

    all_sizes_are_gt_0 = all([sz > 0 for sz in split_sizes.values()])
    split_sizes_eq_dataset_size = sum(split_sizes.values()) == len(dataset)  # type: ignore
    if not (all_sizes_are_gt_0 and split_sizes_eq_dataset_size):
        raise ValueError(
            f"could not create valid dataset split sizes: {split_sizes}, "
            f"full dataset size is {len(dataset)}"  # type: ignore
        )

    # YUCK! Want a map from the dataset designation to teh set itself, but "random_split" takes a list
    # of lengths of dataset. So we do this verbose rigamarol.
    return dict(
        zip(
            keys,
            random_split(
                dataset,
                [split_sizes[k] for k in keys],
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
) -> Dict[str, DataLoader]:
    split_datasets = get_datasets(
        dataset_descriptor_file,
        Sx,
        Sy,
        split_fractions_override=split_fractions_override,
        normalize_images=normalize_images,
    )
    augmentations: List[DualInputModule] = (
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