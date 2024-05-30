import os

import torch
import warnings

from tqdm import tqdm
from functools import partial

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset, random_split


from typing import Any, List, Dict, Optional, Tuple, MutableMapping, Iterable, Union

from yogo.data.blobgen import BlobDataset
from yogo.data.utils import collate_batch_robust
from yogo.data.split_fractions import SplitFractions
from yogo.data.yogo_dataset import ObjectDetectionDataset
from yogo.data.dataset_definition_file import DatasetDefinition
from yogo.data.data_transforms import (
    DualInputModule,
    RandomHorizontalFlipWithBBs,
    RandomVerticalFlipWithBBs,
    MultiArgSequential,
)


def guess_suggested_num_workers() -> Optional[int]:
    """
    It turns out that it is tricky to figure out the number of CPUs across
    different computers. The snippet for suggested number of workers is
    copied from PyTorch.

    https://github.com/pytorch/pytorch/blob/ \
            1f845d589885311447e6021def9da2463c8a989e/ \
            torch/utils/data/dataloader.py#L534-L548
    """

    # try to compute a suggested max number of worker based on system's resource
    max_num_worker_suggest = None
    if hasattr(os, "sched_getaffinity"):
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))  # type: ignore
        except Exception:
            pass

    if max_num_worker_suggest is None:
        # os.cpu_count() could return Optional[int]
        # get cpu count first and check None in order to satisfy mypy check
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            max_num_worker_suggest = cpu_count

    if max_num_worker_suggest is None:
        warnings.warn("could not figure out the number of cpus on this machine")
        return None

    return max_num_worker_suggest


def choose_dataloader_num_workers(
    dataset_size: int, requested_num_workers: Optional[int] = None
) -> int:
    if dataset_size < 1000:
        return 0
    elif requested_num_workers is not None:
        return requested_num_workers
    else:
        return min(guess_suggested_num_workers() or 32, 64)


def get_datasets(
    dataset_definition: DatasetDefinition,
    Sx: int,
    Sy: int,
    rgb: bool = False,
    image_hw: Tuple[int, int] = (772, 1032),
    normalize_images: bool = False,
    split_fraction_override: Optional[SplitFractions] = None,
) -> MutableMapping[str, Dataset[Any]]:
    """
    The job of this function is to convert the dataset_definition_file to actual pytorch datasets.

    See the spec in `dataset_definition_file.py`
    """
    full_dataset: ConcatDataset[ObjectDetectionDataset] = ConcatDataset(
        ObjectDetectionDataset(
            dsp.image_path,
            dsp.label_path,
            Sx,
            Sy,
            image_hw=image_hw,
            rgb=rgb,
            classes=dataset_definition.classes,
            normalize_images=normalize_images,
        )
        for dsp in tqdm(dataset_definition.dataset_paths, desc="loading dataset")
    )

    if (
        dataset_definition.test_dataset_paths is not None
        and len(dataset_definition.test_dataset_paths) > 0
    ):
        test_dataset: ConcatDataset[ObjectDetectionDataset] = ConcatDataset(
            ObjectDetectionDataset(
                dsp.image_path,
                dsp.label_path,
                Sx,
                Sy,
                image_hw=image_hw,
                rgb=rgb,
                classes=dataset_definition.classes,
                normalize_images=normalize_images,
            )
            for dsp in tqdm(
                dataset_definition.test_dataset_paths, desc="loading test dataset"
            )
        )
        if split_fraction_override is not None:
            split_datasets = split_dataset(
                ConcatDataset([full_dataset, test_dataset]), split_fraction_override
            )
        else:
            assert "test" not in dataset_definition.split_fractions
            split_datasets = {
                **split_dataset(full_dataset, dataset_definition.split_fractions),
                "test": test_dataset,
            }
    else:
        if split_fraction_override is not None:
            split_datasets = split_dataset(full_dataset, split_fraction_override)
        else:
            split_datasets = split_dataset(
                full_dataset, dataset_definition.split_fractions
            )

    if dataset_definition.thumbnail_augmentation is not None:
        for k, v in dataset_definition.thumbnail_augmentation.items():
            if not isinstance(v, list):
                dataset_definition.thumbnail_augmentation[k] = [v]

        bd = BlobDataset(
            dataset_definition.thumbnail_augmentation,  # type: ignore
            Sx=Sx,
            Sy=Sy,
            classes=dataset_definition.classes,
            n=100,
            length=len(split_datasets["train"]) // 2,  # type: ignore
            background_img_shape=image_hw,
            normalize_images=normalize_images,
        )
        split_datasets["train"] = ConcatDataset([split_datasets["train"], bd])

    return split_datasets


def split_dataset(
    dataset: Dataset, split_fractions: SplitFractions
) -> MutableMapping[str, Dataset[Any]]:
    if not hasattr(dataset, "__len__"):
        raise ValueError(
            f"dataset {dataset} must have a length (specifically, `__len__` must be defined)"
        )

    keys = split_fractions.keys()
    partition_sizes = split_fractions.partition_sizes(len(dataset))

    # YUCK! Want a map from the dataset designation to the set itself, but "random_split" takes a list
    # of lengths of dataset. So we do this verbose rigamarole.
    return dict(
        zip(
            keys,
            random_split(
                dataset,
                [partition_sizes[k] for k in keys],
                generator=torch.Generator().manual_seed(7271978),
            ),
        )
    )


def get_dataloader(
    dataset_definition: DatasetDefinition,
    batch_size: int,
    Sx: int,
    Sy: int,
    training: bool = True,
    image_hw: Tuple[int, int] = (772, 1032),
    rgb: bool = False,
    normalize_images: bool = False,
    split_fraction_override: Optional[SplitFractions] = None,
) -> Dict[str, DataLoader]:
    split_datasets = get_datasets(
        dataset_definition,
        Sx,
        Sy,
        rgb=rgb,
        image_hw=image_hw,
        normalize_images=normalize_images,
        split_fraction_override=split_fraction_override,
    )

    augmentations: List[DualInputModule] = (
        [
            RandomHorizontalFlipWithBBs(0.5),
            RandomVerticalFlipWithBBs(0.5),
        ]
        if training
        else []
    )

    # if ddp hasn't been initialized, these will raise a
    # RuntimeError instead of returning 0 and 1 respectively.
    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    except RuntimeError:
        rank = 0
        world_size = 1

    d = dict()
    for designation, dataset in split_datasets.items():
        # catch case of len(dataset) == 0
        dataset_len = len(dataset)  # type: ignore
        if dataset_len == 0:
            continue

        augs = augmentations if designation == "train" else []

        d[designation] = _get_dataloader(
            dataset,
            batch_size=batch_size,
            augmentations=augs,
            rank=rank,
            world_size=world_size,
        )
    return d


def _get_dataloader(
    dataset: Dataset,
    batch_size: int,
    augmentations: List[DualInputModule],
    rank: int,
    world_size: int,
) -> DataLoader:
    transforms = MultiArgSequential(*augmentations)

    sampler: Iterable = DistributedSampler(
        dataset,
        rank=rank,
        num_replicas=world_size,
    )

    # TODO this division by world_size is hacky. Starting up the dataloaders
    # are in*sane*ly slow. This helps reduce the problem, but tbh not by much
    dataset_len = len(dataset)  # type: ignore
    num_workers = max(1, choose_dataloader_num_workers(dataset_len) // world_size)

    return DataLoader(
        dataset,
        shuffle=False,
        sampler=sampler,
        drop_last=False,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(7271978),
        collate_fn=partial(collate_batch_robust, transforms=transforms),
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


DATALOADER_TYPES = Union[
    DataLoader[ConcatDataset[ObjectDetectionDataset]],
    DataLoader[Subset[ConcatDataset[ObjectDetectionDataset]]],
]


DATASET_TYPES = Union[
    ObjectDetectionDataset,
    Subset[ConcatDataset[ObjectDetectionDataset]],
]


def get_class_counts(
    d: DATALOADER_TYPES, num_classes: int, verbose: bool = True
) -> torch.Tensor:
    """
    d is a Dataloader of one ConcatDataset of ObjectDetectionDatasets and BlobGen datasets.
    This function should iterate through the datasets of d, ignore BlobDataset datasets,
    and sum the (num_classes,) tensors returned by `calc_class_counts` of ObjectDetectionDatasets

    it's sort of a weird tree-like structure. from `get_dataloader(path_to_auggd_data)['train']`, you
    get a Dataloader[ConcatDataset], and the ConcatDataset has [ConcatDataset, BlobGen], and the inner
    ConcatDataset is the concat of all ObjectDetectionDatasets. So just traverse the tree, adding the
    datasets in ConcatDatasets, calculating the class counts in each.
    """
    class_counts = torch.zeros(num_classes, dtype=torch.long)

    pbar = tqdm(
        total=len(d) * (d.batch_size or 1), desc="counting...", disable=not verbose
    )
    for _, labels in d:  #:
        bs, pd, Sy, Sx = labels.shape
        labels = labels.permute(1, 0, 2, 3)
        labels = labels.reshape(pd, bs * Sy * Sx)
        labels = labels[:, labels[0, :] == 1].long()
        class_counts += torch.bincount(labels[5, :], minlength=num_classes)
        pbar.update(bs)

    return class_counts


def get_image_count(d: DATALOADER_TYPES) -> int:
    s = 0
    if isinstance(d.dataset, ConcatDataset):
        s += d.dataset.cumulative_sizes[-1]
    elif isinstance(d.dataset, Subset):
        s += len(d.dataset)
    else:
        raise TypeError(f"unknown type {type(d.dataset)}")
    return s
