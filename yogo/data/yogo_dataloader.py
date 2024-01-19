import os

import torch
import warnings

from tqdm import tqdm
from pathlib import Path
from functools import partial

from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split

from typing import Any, List, Dict, Tuple, Optional, MutableMapping, Iterable

from yogo.data.blobgen import BlobDataset
from yogo.data.split_fractions import SplitFractions
from yogo.data.yogo_dataset import ObjectDetectionDataset
from yogo.data.dataset_definition_file import DatasetDefinition
from yogo.data.data_transforms import (
    DualInputModule,
    RandomHorizontalFlipWithBBs,
    RandomVerticalFlipWithBBs,
    MultiArgSequential,
)
from torch.utils.data.distributed import DistributedSampler


def guess_suggested_num_workers() -> Optional[int]:
    """
    It turns out that it is tricky to figure out the number of CPUs across different
    computers. I kept on getting warnings that I was provisioning too many, even though
    I was trying to provision mp.cpu_count() workers.

    So, I just copied the code from Pytorch, figuring that they knew how to get it. It's
    not as easy as I thought it'd be. Funky funky funky

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
    dataset_definition_file: str,
    Sx: int,
    Sy: int,
    normalize_images: bool = False,
) -> MutableMapping[str, Dataset[Any]]:
    dataset_definition = DatasetDefinition.from_yaml(Path(dataset_definition_file))
    full_dataset: ConcatDataset[ObjectDetectionDataset] = ConcatDataset(
        ObjectDetectionDataset(
            dsp.image_path,
            dsp.label_path,
            Sx,
            Sy,
            normalize_images=normalize_images,
        )
        for dsp in tqdm(dataset_definition.dataset_paths, desc="loading dataset")
    )

    if dataset_definition.test_dataset_paths is not None:
        test_dataset: ConcatDataset[ObjectDetectionDataset] = ConcatDataset(
            ObjectDetectionDataset(
                dsp.image_path,
                dsp.label_path,
                Sx,
                Sy,
                normalize_images=normalize_images,
            )
            for dsp in tqdm(
                dataset_definition.test_dataset_paths, desc="loading test dataset"
            )
        )
        split_datasets: MutableMapping[str, Dataset[Any]] = {
            "train": full_dataset,
            **split_dataset(test_dataset, dataset_definition.split_fractions),
        }
    else:
        split_datasets = split_dataset(full_dataset, dataset_definition.split_fractions)

    # hardcode the blob agumentation for now
    # this should be moved into the dataset description file
    if dataset_definition.thumbnail_augmentation is not None:
        # some issue w/ Dict v Mapping TODO come back to this
        bd = BlobDataset(
            dataset_definition.thumbnail_augmentation,  # type: ignore
            Sx=Sx,
            Sy=Sy,
            n=8,
            length=len(split_datasets["train"]),  # type: ignore
            blend_thumbnails=True,
            thumbnail_sigma=2,
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


def collate_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    transforms: MultiArgSequential = MultiArgSequential(),
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    normalize_images: bool = False,
) -> Dict[str, DataLoader]:
    split_datasets = get_datasets(
        dataset_descriptor_file,
        Sx,
        Sy,
        normalize_images=normalize_images,
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

        transforms = MultiArgSequential(
            *augmentations if designation == "train" else [],
        )

        sampler: Iterable = DistributedSampler(
            dataset,
            rank=rank,
            num_replicas=world_size,
        )

        # TODO this division by world_size is hacky. Starting up the dataloaders
        # are in*sane*ly slow. This helps reduce the problem, but tbh not by much
        num_workers = max(1, choose_dataloader_num_workers(dataset_len) // world_size)

        d[designation] = DataLoader(
            dataset,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            pin_memory=True,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            generator=torch.Generator().manual_seed(7271978),
            collate_fn=partial(collate_batch, transforms=transforms),
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )
    return d


def get_class_counts(d: DataLoader[ConcatDataset], num_classes: int) -> torch.Tensor:
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
    dset_iters = [d.dataset.datasets]  # type:ignore
    for dset_iter in dset_iters:
        for dataset in tqdm(dset_iter, desc="calculating class weights"):
            if isinstance(dataset, ConcatDataset):
                dset_iters.append(dataset.datasets)
            elif isinstance(dataset, ObjectDetectionDataset):
                class_counts += dataset.calc_class_counts()

    if class_counts is None:
        raise ValueError("could not find any ObjectDetectionDatasets in ConcatDataset")

    return class_counts


def normalized_inverse_frequencies(d: List[int]) -> torch.Tensor:
    t = torch.tensor(d)
    class_freq = t / t.sum()
    class_weights = 1.0 / class_freq
    class_weights = class_weights / class_weights.sum()
    return class_weights
