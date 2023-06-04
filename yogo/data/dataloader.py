import os
import torch

from tqdm import tqdm
from functools import partial

from torchvision.transforms import Resize, RandomAdjustSharpness, ColorJitter
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split

from typing import List, Dict, Tuple, Optional

from yogo.data.blobgen import BlobDataset
from yogo.data.dataset import ObjectDetectionDataset
from yogo.data.dataset_description_file import load_dataset_description
from yogo.data.data_transforms import (
    DualInputModule,
    DualInputId,
    RandomHorizontalFlipWithBBs,
    RandomVerticalFlipWithBBs,
    RandomVerticalCrop,
    ImageTransformLabelIdentity,
    MultiArgSequential,
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
        thumbnail_augmentations,
    ) = load_dataset_description(dataset_description_file)

    # can we speed this up? multiproc dataset creation?
    full_dataset: ConcatDataset[ObjectDetectionDataset] = ConcatDataset(
        ObjectDetectionDataset(
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

    split_datasets = split_dataset(full_dataset, split_fractions)

    # hardcode the blob agumentation for now
    # this should be moved into the dataset description file
    if thumbnail_augmentations is not None:
        bd = BlobDataset(
            thumbnail_augmentations,
            Sx=Sx,
            Sy=Sy,
            n=8,
            length=len(split_datasets["train"]) // 10,  # type: ignore
            blend_thumbnails=True,
            thumbnail_sigma=2,
            normalize_images=normalize_images,
        )
        split_datasets["train"] = ConcatDataset([split_datasets["train"], bd])

    return split_datasets


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


def collate_batch(batch, transforms):
    images, labels = zip(*batch)

    batched_images = torch.stack(images)

    num_labels_per_img = [len(label) for label in labels]
    batch_size= len(labels)
    num_labels = 1 + max(num_labels_per_img)
    labels_w  = len(labels[0][0])
    batched_labels = torch.zeros((batch_size, num_labels, labels_w))

    for i, (label_size, label) in enumerate(zip(num_labels_per_img, labels)):
        batched_labels[i, 0, 0] = label_size
        batched_labels[i, 1:1+label_size, :] = label

    # return transforms(batched_images, batched_labels)
    return batched_images, batched_labels


def get_dataloader(
    dataset_descriptor_file: str,
    batch_size: int,
    Sx: int,
    Sy: int,
    training: bool = True,
    preprocess_type: Optional[str] = None,
    vertical_crop_size: Optional[float] = None,
    resize_shape: Optional[Tuple[int, int]] = None,
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
        image_preprocess = Resize(resize_shape, antialias=True)
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
            collate_fn=partial(collate_batch, transforms=transforms),
        )
    return d
