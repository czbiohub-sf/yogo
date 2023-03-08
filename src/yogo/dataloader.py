import os
import csv
import yaml
import torch

from pathlib import Path
from functools import partial
from collections import defaultdict

import torchvision.ops as ops

from torchvision import datasets
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize, RandomAdjustSharpness, ColorJitter
from torch.utils.data import ConcatDataset, DataLoader, random_split, Subset

from typing import Any, List, Dict, Union, Tuple, Optional, Callable, Literal, cast

from yogo.data_transforms import (
    DualInputModule,
    DualInputId,
    RandomHorizontalFlipWithBBs,
    RandomVerticalFlipWithBBs,
    RandomVerticalCrop,
    ImageTransformLabelIdentity,
    MultiArgSequential,
)


YOGO_CLASS_ORDERING = [
    "healthy",
    "ring",
    "trophozoite",
    "schizont",
    "gametocyte",
    "wbc",
    "misc",
]


DatasetSplitName = Literal["train", "val", "test"]


def count_dataloader_class(dataloader, class_index: int) -> int:
    s = 0
    for _, labels in dataloader:
        s += sum((l[:, 0] == class_index).sum().item() for l in labels if len(l) > 0)
    return s


def get_class_counts_for_dataloader(dataloader, class_names):
    return {c: count_dataloader_class(dataloader, i) for i, c in enumerate(class_names)}


def read_grayscale(img_path):
    try:
        return read_image(str(img_path), ImageReadMode.GRAY)
    except RuntimeError as e:
        raise RuntimeError(f"file {img_path} threw: {e}")


def collate_batch(batch, device="cpu", transforms=None):
    # perform image transforms here so we can transform in batches! :)
    inputs, labels = zip(*batch)
    batched_inputs = torch.stack(inputs)
    batched_labels = torch.stack(labels)
    return transforms(
        batched_inputs.to(device, non_blocking=True),
        batched_labels.to(device, non_blocking=True),
    )


def split_labels_into_bins(
    labels: torch.Tensor, Sx, Sy
) -> Dict[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    # it is really a single-element long tensor
    d: Dict[Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]] = defaultdict(list)
    for label in labels:
        i = torch.div(label[1], (1 / Sx), rounding_mode="trunc").long()
        j = torch.div(label[2], (1 / Sy), rounding_mode="trunc").long()
        d[(i, j)].append(label)
    return {k: torch.vstack(vs) for k, vs in d.items()}


def format_labels(
    labels: torch.Tensor,
    Sx: int,
    Sy: int,
    num_classes: int,
) -> torch.Tensor:
    """
    input:
        Sx: int,
        Sy: int,
        label_batch: List[torch.Tensor], and len(label_batch) == batch_size
        num_classes: int
    output:
        torch.Tensor of shape (batch_size, masked_label_len, Sy, Sx)

    dimension masked_label is [mask, xc, yc, w, h, *classes], where mask == 1
    if there is a label associated with (Sy,Sx) at the given batch, else 0. If
    mask is 0, then the rest of the label values are "don't care" values (just
    setting to 0 is fine).

    TODO: maybe we can drop some sync points by converting label_batch to tensor?
    Have a parameter for "num labels" or smth, and have all tensors be the size
    of the minimum tensor size (instead of having a list)
    """
    with torch.no_grad():
        output = torch.zeros(1 + num_classes + 1, Sy, Sx)
        label_cells = split_labels_into_bins(labels, Sx, Sy)

        for (k, j), cell_label in label_cells.items():
            pred_square_idx = 0  # this is a remnant of Sx,Sy being small; remove?
            output[0, j, k] = 1
            output[1:5, j, k] = cell_label[pred_square_idx][1:]
            output[5, j, k] = cell_label[pred_square_idx][0]

        return output


def load_labels_from_path(label_path: Path, dataset_classes: List[str], Sx: int, Sy: int) -> List[List[float]]:

    "loads labels from label file, given by image path"
    labels: List[List[float]] = []
    try:
        with open(label_path, "r") as f:
            file_chunk = f.read(1024)
            f.seek(0)

            try:
                dialect = csv.Sniffer().sniff(file_chunk)
                has_header = csv.Sniffer().has_header(file_chunk)
                reader = csv.reader(f, dialect)
            except csv.Error:
                # emtpy file, no labels, just keep moving
                return labels

            if has_header:
                next(reader, None)

            for row in reader:
                assert (
                    len(row) == 5
                ), f"should have [class,xc,yc,w,h] - got length {len(row)} {row}"

                """
                dataset_classes is the ordering of classes that are given by
                label-studio. So we get the class of the prediction from
                `int(row[0])`, and get the index of that from YOGO_CLASS_ORDERING
                """
                if row[0].isnumeric():
                    label_idx = YOGO_CLASS_ORDERING.index(dataset_classes[int(row[0])])
                else:
                    label_idx = YOGO_CLASS_ORDERING.index(row[0])

                # float for everything so we can make tensors of labels
                labels.append([float(label_idx)] + [float(v) for v in row[1:]])
    except FileNotFoundError:
        pass

    labels = torch.Tensor(labels)
    labels[:, 1:] = ops.box_convert(labels[:, 1:], "cxcywh", "xyxy")
    return format_labels(labels, Sx, Sy, len(classes))


class ObjectDetectionDataset(datasets.VisionDataset):
    def __init__(
        self,
        dataset_classes: List[str],
        image_path: Path,
        label_path: Path,
        Sx,
        Sy,
        loader: Callable = read_grayscale,
        extensions: Optional[Tuple[str]] = ("png",),
        is_valid_file: Optional[Callable[[str], bool]] = None,
        *args,
        **kwargs,
    ):
        # the super().__init__ just sets transforms
        # the image_path is just for repr
        super().__init__(str(image_path), *args, **kwargs)

        self.classes = YOGO_CLASS_ORDERING
        self.image_folder_path = image_path
        self.label_folder_path = label_path
        self.loader = loader

        self.samples = self.make_dataset(
        Sx, Sy,
            is_valid_file=is_valid_file, extensions=extensions, dataset_classes=dataset_classes
        )

    def make_dataset(
        self,
        Sx: int,
        Sy: int,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        dataset_classes: List[str] = YOGO_CLASS_ORDERING,
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
        for label_file_path in self.label_folder_path.glob("*"):
            image_paths =  [
                self.image_folder_path / label_file_path.with_suffix(sfx).name
                for sfx in [".png", ".jpg"]
            ]

            try:
                image_file_path = next(
                    ip for ip in image_paths
                    if (
                        ip.exists() and is_valid_file(str(ip))
                    )
                )
            except StopIteration as e:
                # raise exception here? logic being that we want to know very quickly that we don't have
                # all the labels we need. Open to changes, though.
                raise FileNotFoundError(
                    f"None of the following images exist: {image_paths}"
                ) from e
            labels = load_labels_from_path(label_file_path, dataset_classes, Sx, Sy)
            samples.append((str(image_file_path), labels))

        return samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[List[float]]]:
        """From torchvision.datasets.folder.DatasetFolder
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


def load_dataset_description(
    dataset_description: str,
) -> Tuple[List[str], List[Dict[str, Path]], Dict[str, float]]:
    with open(dataset_description, "r") as desc:
        with open(dataset_description, "r"):
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


def get_datasets(
    dataset_description_file: str,
    batch_size: int,
    Sx,
    Sy,
    training: bool = True,
    split_fractions_override: Optional[Dict[str, float]] = None,
) -> Dict[DatasetSplitName, Subset[ConcatDataset[ObjectDetectionDataset]]]:
    (
        dataset_classes,
        dataset_paths,
        split_fractions,
    ) = load_dataset_description(dataset_description_file)

    full_dataset: ConcatDataset[ObjectDetectionDataset] = ConcatDataset(
        ObjectDetectionDataset(
            dataset_classes,
            dataset_desc["image_path"],
            dataset_desc["label_path"],
            Sx,
            Sy
        )
        for dataset_desc in dataset_paths
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
                generator=torch.Generator().manual_seed(101010),
            ),
        )
    )


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
) -> Dict[DatasetSplitName, DataLoader]:
    split_datasets = get_datasets(
        dataset_descriptor_file,
        batch_size,
        Sx,
        Sy,
        training=training,
        split_fractions_override=split_fractions_override,
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
            image_preprocess, *augmentations if designation == "train" else [],
        )
        d[designation] = DataLoader(
            dataset,
            shuffle=True,
            drop_last=False,
            batch_size=batch_size,
            persistent_workers=True,  # why would htis not be on by default lol
            multiprocessing_context="spawn",
            num_workers=len(os.sched_getaffinity(0)) // 2,  # type: ignore
            generator=torch.Generator().manual_seed(101010),
            collate_fn=partial(collate_batch, device=device, transforms=transforms),
        )
    return d
