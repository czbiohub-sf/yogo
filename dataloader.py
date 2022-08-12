import os
import torch

from torchvision import datasets
from torchvision.io import read_image
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)

from typing import List, Dict, Union, Tuple, Optional, Callable


class ImageFolderWithLabels(datasets.ImageFolder):
    """ImageFolder with minor modifications to make training/use easier

    Changes:
        - save the idx_to_class in the instance so the target_transform to folder name is quick
        - `sample_from_class` method that quickly gives a sample of a given class of a given size
        - exclude_classes kwarg for excluding class subsets
    """

    def __init__(self, *args, exclude_classes: List[Union[int, str]] = [], **kwargs):
        self.exclude_classes = [str(excl) for excl in exclude_classes]

        super().__init__(*args, **kwargs)

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        def target_transform(idx):
            return int(self.idx_to_class[idx])

        self.target_transform = target_transform

        # for `sample_from_class`
        self.class_to_samples: Dict[int, List[str]] = dict()
        for el, label in self.imgs:
            el_class = target_transform(label)
            if el_class in self.class_to_samples:
                self.class_to_samples[el_class].append(el)
            else:
                self.class_to_samples[el_class] = [el]

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        "Adapted from torchvision.datasets.folder.py"
        classes = sorted(
            entry.name
            for entry in os.scandir(directory)
            if entry.is_dir() and entry.name not in self.exclude_classes
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def sample_from_class(self, clss, count):
        # TODO: probably a better way to do this
        sample_set = self.class_to_samples[clss]
        idxs = torch.randint(len(sample_set), [count, 1])
        samples = []
        for idx in idxs:
            T = self.transform(self.loader(sample_set[idx]))
            T = torch.unsqueeze(T, 0)
            samples.append(T)
        return torch.cat(samples)


def get_dataset(
    root_dir: str,
    batch_size: int,
    split_percentages: List[float] = [1],
    exclude_classes: List[str] = [],
    training: bool = True,
):
    assert (
        sum(split_percentages) == 1
    ), f"split_percentages must add to 1 - got {split_percentages}"

    augmentations = (
        [RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5)] if training else []
    )
    transforms = Compose([Resize([150, 200]), *augmentations])
    full_dataset = ImageFolderWithLabels(
        root=root_dir,
        transform=transforms,
        loader=read_image,
        exclude_classes=exclude_classes,
    )

    first_split_sizes = [
        int(rel_size * len(full_dataset)) for rel_size in split_percentages[:-1]
    ]
    final_split_size = [len(full_dataset) - sum(first_split_sizes)]
    split_sizes = first_split_sizes + final_split_size

    assert all([sz > 0 for sz in split_sizes]) and sum(split_sizes) == len(
        full_dataset
    ), f"could not create valid dataset split sizes: {split_sizes}, full dataset size is {len(full_dataset)}"

    return random_split(full_dataset, split_sizes)


def get_dataloader(
    root_dir: str,
    batch_size: int,
    split_percentages: List[float] = [1],
    exclude_classes: List[str] = [],
    training: bool = True,
):
    split_datasets = get_dataset(
        root_dir, batch_size, split_percentages, exclude_classes, training=training
    )
    return [
        DataLoader(split_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        for split_dataset in split_datasets
    ]
