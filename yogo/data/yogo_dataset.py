import csv
import json
import torch
import numpy as np

from torchvision import ops
from torchvision import datasets
from torchvision.transforms import Resize

from pathlib import Path
from functools import partial
from typing import List, Dict, Union, Tuple, Optional, Callable, Any, cast

from yogo.data.utils import read_image_robust


LABEL_TENSOR_PRED_DIM_SIZE = 1 + 4 + 1

# Guess: 200 sq px is probably about OK
# FIXME: hard-coded for YOGO
AREA_FILTER_THRESHOLD = 200 / (772 * 1032)


def format_labels_tensor(labels: torch.Tensor, Sx: int, Sy: int) -> torch.Tensor:
    """
    converts a tensor of shape (N, 5) into a tensor of shape (LABEL_TENSOR_PRED_DIM, Sy, Sx)

    The input tensor `labels` has elements (class_idx, x, y, x, y), where xs and ys are
    normalized to the input image's shape.

    The LABEL_TENSOR_PRED_DIM consists of (mask, x, y, x, y, class_idx).
    The mask is 1 if there is a prediction, 0 otherwise.
    """
    output = torch.zeros(LABEL_TENSOR_PRED_DIM_SIZE, Sy, Sx)

    # find the center of each bbox in a grid of size (Sx,Sy)
    iis = (labels[:, 1] + labels[:, 3]) * Sx // 2
    jjs = (labels[:, 2] + labels[:, 4]) * Sy // 2

    # Go through each of our labels and
    for i, j, label in zip(iis.int(), jjs.int(), labels):
        output[0, j, i] = 1  # mask that there is a prediction here
        output[1:5, j, i] = label[1:]  # xyxy
        output[5, j, i] = label[0]  # class prediction idx

    return output


def correct_label_idx(
    label: str,
    classes: List[str],
    notes_data: Optional[Dict[str, Any]] = None,
) -> int:
    if notes_data is None:
        # this is the best we can do
        return int(label)
    elif label.isnumeric():
        label_name: Optional[str] = None
        for row in notes_data["categories"]:
            if int(label) == int(row["id"]):
                label_name = row["name"]
                break

        if label_name is None:
            raise ValueError(f"label index {label} not found in notes.json file")

        return classes.index(label_name)
    else:
        return classes.index(label)


def load_labels(
    label_path: Path,
    classes: List[str],
    notes_data: Optional[Dict[str, Any]] = None,
) -> List[List[float]]:
    "loads labels from label file, given by image path"
    labels: List[List[float]] = []

    with open(label_path, "r") as f:
        file_chunk = f.read(1024)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(file_chunk)
            has_header = csv.Sniffer().has_header(file_chunk)
            reader = csv.reader(f, dialect)
        except csv.Error:
            # empty file, no labels, just keep moving
            return []

        if has_header:
            next(reader, None)

        for row in reader:
            assert (
                len(row) == 5
            ), f"should have [class,xc,yc,w,h] - got length {len(row)} {row}"

            xc, yc, w, h = map(float, row[1:])

            if w * h < AREA_FILTER_THRESHOLD:
                continue

            label_idx = correct_label_idx(row[0], classes, notes_data)

            # float for everything so we can make tensors of labels
            labels.append([float(label_idx), xc, yc, w, h])

    return labels


def label_file_to_tensor(
    label_path: Path,
    Sx: int,
    Sy: int,
    classes: List[str],
    notes_data: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    "loads labels from label file into a tensor suitible for back prop, given by image path"

    try:
        labels = load_labels(label_path, classes=classes, notes_data=notes_data)
    except Exception as e:
        raise RuntimeError(f"exception from {label_path}") from e

    labels_tensor = torch.Tensor(labels)

    if labels_tensor.nelement() == 0:
        return torch.zeros(LABEL_TENSOR_PRED_DIM_SIZE, Sy, Sx)

    labels_tensor[:, 1:] = ops.box_convert(labels_tensor[:, 1:], "cxcywh", "xyxy")
    return format_labels_tensor(labels_tensor, Sx, Sy)


class ObjectDetectionDataset(datasets.VisionDataset):
    def __init__(
        self,
        image_folder_path: Path,
        label_folder_path: Path,
        Sx,
        Sy,
        classes: List[str],
        image_hw: Tuple[int, int] = (772, 1032),
        rgb: bool = False,
        normalize_images: bool = False,
        extensions: Tuple[str, ...] = ("png", "jpg", "jpeg", "tif"),
        is_valid_file: Optional[Callable[[str], bool]] = None,
        *args,
        **kwargs,
    ):
        # the super().__init__ just sets transforms
        # the image_path is just for repr
        super().__init__(str(image_folder_path), *args, **kwargs)

        self.classes = classes
        self.image_folder_path = image_folder_path
        self.label_folder_path = label_folder_path
        self.loader = partial(read_image_robust, retries=3, min_duration=0.1, rgb=rgb)
        self.resize = Resize(image_hw, antialias=True)
        self.normalize_images = normalize_images
        self.notes_data: Optional[Dict[str, Any]] = None

        # https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # essentially, to avoid dataloader workers from copying tonnes of mem,
        # we can't store samples in lists. Hence, numpy arrays.
        image_paths, label_paths = self.make_dataset(
            Sx,
            Sy,
            is_valid_file=is_valid_file,
            extensions=extensions,
        )

        self.Sx = Sx
        self.Sy = Sy

        self._image_paths = np.array(image_paths).astype(np.unicode_)
        self._label_paths = np.array(label_paths).astype(np.unicode_)

    def make_dataset(
        self,
        Sx: int,
        Sy: int,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> Tuple[List[str], List[str]]:
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

        if (self.label_folder_path.parent / "notes.json").exists():
            with open(str(self.label_folder_path.parent / "notes.json"), "r") as notes:
                self.notes_data = json.load(notes)

        # maps file name to a list of tuples of bounding boxes + classes
        image_paths: List[str] = []
        label_paths: List[str] = []
        missing_images: List[str] = []

        for label_file_path in self.label_folder_path.glob("*.txt"):
            # ignore (*nix convention) hidden files
            if label_file_path.name.startswith("."):
                continue

            possible_image_paths = [
                self.image_folder_path / label_file_path.with_suffix(sfx).name
                for sfx in [".png", ".jpg"]
            ]

            try:
                image_file_path = next(
                    ip
                    for ip in possible_image_paths
                    if (ip.exists() and is_valid_file(str(ip)))
                )
                image_paths.append(str(image_file_path))
                label_paths.append(str(label_file_path))
            except StopIteration:
                # image is missing
                missing_images.append(str(label_file_path))
                if len(image_paths) > 10:
                    # just give up!
                    break

        if len(missing_images) > 0:
            if len(missing_images) < 5:
                missing_subset = missing_images
                list_message = " "
            else:
                missing_subset = missing_images[:3]
                list_message = " a sample of "

            raise FileNotFoundError(
                f"{'at least ' if len(missing_images) == 10 else ' '}{len(missing_images)}"
                f"images not found in {self.image_folder_path}; "
                f"({len(image_paths)} images were found). Here's{list_message}the list:\n"
                f"{missing_subset}"
            )

        return image_paths, label_paths

    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        image_path = self._image_paths[index]
        label_path = self._label_paths[index]

        maybe_image = self.loader(image_path)
        if maybe_image is None:
            return None

        image = self.resize(maybe_image)

        labels = label_file_to_tensor(
            Path(label_path), self.Sx, self.Sy, self.classes, self.notes_data
        )

        if self.normalize_images:
            # turns our torch.uint8 tensor 'sample' into a torch.FloatTensor
            image = image / 255

        return image, labels

    def __len__(self) -> int:
        return len(self._image_paths)

    def calc_class_counts(self) -> torch.Tensor:
        """
        returns a tensor of shape (num_classes,) where each index is the number of
        times that class appears in the dataset
        """
        class_counts = torch.zeros(len(self.classes), dtype=torch.long)
        for label_path in self._label_paths:
            labels = load_labels(
                label_path, classes=self.classes, notes_data=self.notes_data
            )
            for label in labels:
                class_counts[int(label[0])] += 1
        return class_counts
