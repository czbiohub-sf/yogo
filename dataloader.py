import os
import csv
import yaml
import torch

from pathlib import Path

from torchvision import datasets
from torchvision.io import read_image
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)

from typing import List, Dict, Union, Tuple, Optional, Callable, cast


class ObjectDetectionDataset(datasets.DatasetFolder):
    """TODO FIXME

    I think the abstractions are wrong here. Subclassing DatasetFolder
    is definitely wrong (we do too much hacky stuff to work around it,
    we should just subclass DatasetFolders superclass), and the format
    of labels is most likely wrong.

    TLDR i made some bad decisions, but by working through them I think
    that I know the right way to go about this.
    """

    def __init__(self, dataset_description, *args, **kwargs):
        (
            classes,
            image_path,
            label_path,
            split_fractions,
        ) = self.load_dataset_description(dataset_description)
        self.classes = classes
        self.image_folder_path = image_path
        self.label_folder_path = label_path
        self.split_fractions = split_fractions
        super().__init__(
            image_path, loader=datasets.folder.default_loader, *args, **kwargs
        )

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, List[List[float]]]]:
        """
        torchvision.datasets.folder.make_dataset doc string states:
            "Generates a list of samples of a form (path_to_sample, class)"

        This is designed for a dataset for classficiation (that is, mapping
        image to class), where we have a dataset for object detection (image
        to list of bounding boxes). datasets.DatasetFolder has an image loader
        by default that I would like to use, but if it is all too awkward,
        I'll have to rewrite this subclassing VisionDataset.

        Mostly copied from Pytorch's implementation[0], with changes on how we
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
                return datasets.folder.has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        # maps file name to a list of tuples of bounding boxes + classes
        instances: List[Tuple[str, List[List[float]]]] = []
        for img_file_path in self.images.glob("*"):
            if is_valid_file(str(img_file_path)):
                labels = self.load_labels_from_image_name(img_file_path)
                instances.append((str(img_file_path), labels))
        return instances

    def load_labels_from_image_name(self, image_path: Path) -> List[List[float]]:
        """
        loads labels from label file, given by image path
        """
        labels = []
        label_filename = str(image_path).replace(image_path.suffix, "csv")

        with open(self.label_folder_path / label_filename, "r") as f:
            reader = csv.reader(f)
            has_header = csv.Sniffer().has_header(f.read(1024))
            f.seek(0)
            if has_header:
                next(reader, None)

            for row in reader:
                assert (
                    len(row) == 5
                ), "should have [class,xc,yc,w,h] - got length {len(row)}"
                labels.append([float(v) for v in row])

        return labels

    def load_dataset_description(
        self, dataset_description
    ) -> Tuple[List[str], Path, Path, Dict[str, float]]:
        with open(dataset_description, "r") as desc:
            yaml_data = yaml.safe_load(desc)

            classes = yaml_data["class_names"]
            image_path = Path(yaml_data["image_path"])
            label_path = Path(yaml_data["label_path"])
            split_fractions = {
                k: float(v) for k, v in yaml_data["dataset_split_fractions"].items()
            }

            if not sum(split_fractions.values()) == 1:
                raise ValueError(
                    f"invalid split fractions for dataset: split fractions must add to 1, got {split_fractions}"
                )

            if not (image_path.is_dir() and label_path.is_dir()):
                raise FileNotFoundError(
                    f"image_path or label_path do not lead to a directory\n{image_path=}\n{label_path=}"
                )

            return classes, image_path, label_path, split_fractions


def get_dataset(
    dataset_description: str,
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
    full_dataset = ObjectDetectionDataset(
        dataset_description=dataset_description,
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


if __name__ == "__main__":
    ODL = ObjectDetectionDataset("healthy_cell_dataset.yml")
    for data in ODL:
        print(data)
