import os
import torch

from tqdm import tqdm
from functools import partial

from torchvision.transforms import Resize
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split

from typing import List, Dict, Tuple, Optional, Any, MutableMapping, Iterable

from yogo.data.blobgen import BlobDataset
from torch.utils.data.distributed import DistributedSampler
from yogo.data.yogo_dataset import ObjectDetectionDataset
from yogo.data.dataset_description_file import load_dataset_description
from yogo.data.data_transforms import (
    DualInputModule,
    DualInputId,
    RandomHorizontalFlipWithBBs,
    RandomVerticalFlipWithBBs,
    RandomVerticalCrop,
    MultiArgSequential,
)


def choose_dataloader_num_workers(
    dataset_size: int, requested_num_workers: Optional[int] = None
):
    if dataset_size < 1000:
        return 0
    return (
        requested_num_workers
        if requested_num_workers is not None
        else min(len(os.sched_getaffinity(0)), 64)
    )


def get_datasets(
    dataset_description_file: str,
    Sx: int,
    Sy: int,
    normalize_images: bool = False,
) -> MutableMapping[str, Dataset[Any]]:
    dataset_description = load_dataset_description(dataset_description_file)
    # can we speed this up? multiproc dataset creation?
    full_dataset: ConcatDataset[ObjectDetectionDataset] = ConcatDataset(
        ObjectDetectionDataset(
            dsp["image_path"],
            dsp["label_path"],
            Sx,
            Sy,
            normalize_images=normalize_images,
        )
        for dsp in tqdm(dataset_description.dataset_paths, desc="loading dataset")
    )

    if dataset_description.test_dataset_paths is not None:
        test_dataset: ConcatDataset[ObjectDetectionDataset] = ConcatDataset(
            ObjectDetectionDataset(
                dsp["image_path"],
                dsp["label_path"],
                Sx,
                Sy,
                normalize_images=normalize_images,
            )
            for dsp in tqdm(
                dataset_description.test_dataset_paths, desc="loading test dataset"
            )
        )
        split_datasets: MutableMapping[str, Dataset[Any]] = {
            "train": full_dataset,
            **split_dataset(test_dataset, dataset_description.split_fractions),
        }
    else:
        split_datasets = split_dataset(
            full_dataset, dataset_description.split_fractions
        )

    # hardcode the blob agumentation for now
    # this should be moved into the dataset description file
    if dataset_description.thumbnail_augmentation is not None:
        # some issue w/ Dict v Mapping TODO come back to this
        bd = BlobDataset(
            dataset_description.thumbnail_augmentation,  # type: ignore
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
    dataset: Dataset, split_fractions: Dict[str, float]
) -> MutableMapping[str, Dataset[Any]]:
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
    # so we manually check it. But I am not sure that you can cast to Sized, so mypy complains
    dataset_sizes = {
        k: round(split_fractions[k] * len(dataset)) for k in keys[:-1]  # type: ignore
    }
    final_dataset_size = {keys[-1]: len(dataset) - sum(dataset_sizes.values())}  # type: ignore
    split_sizes = {**dataset_sizes, **final_dataset_size}

    all_sizes_are_gt_0 = all([sz >= 0 for sz in split_sizes.values()])
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

    # if ddp hasn't been initialized, these will raise
    # runtime errors instead of returning 0 and 1 respectively.
    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    except RuntimeError:
        rank = 0
        world_size = 1

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
        # catch case of len(dataset) == 0
        dataset_len = len(dataset)  # type: ignore
        if dataset_len == 0:
            continue

        transforms = MultiArgSequential(
            image_preprocess,
            *augmentations if designation == "train" else [],
        )

        sampler: Iterable = DistributedSampler(
            dataset,
            rank=rank,
            num_replicas=world_size,
        )

        num_workers = choose_dataloader_num_workers(dataset_len)

        d[designation] = DataLoader(
            dataset,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,  # type: ignore
            persistent_workers=num_workers > 0,
            generator=torch.Generator().manual_seed(111111),
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
    datasets in ConcatDatasets, calculating the class counts in
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
