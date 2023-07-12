import csv
import json
import torch
import numpy as np

from pathlib import Path

import torchvision.ops as ops

from torchvision import datasets

from typing import List, Dict, Union, Tuple, Optional, Callable, Any, cast

from yogo.data import YOGO_CLASS_ORDERING
from yogo.data.utils import read_grayscale


LABEL_TENSOR_PRED_DIM_SIZE = 1 + 4 + 1


def format_labels_tensor(labels: torch.Tensor, Sx: int, Sy: int) -> torch.Tensor:
    output = torch.zeros(LABEL_TENSOR_PRED_DIM_SIZE, Sy, Sx)
    iis = (labels[:, 1] + labels[:, 3]) * Sx // 2
    jjs = (labels[:, 2] + labels[:, 4]) * Sy // 2

    for i, j, label in zip(iis.int(), jjs.int(), labels):
        output[0, j, i] = 1  # mask that there is a prediction here
        output[1:5, j, i] = label[1:]  # xyxy
        output[5, j, i] = label[0]  # prediction idx

    return output


def correct_label_idx(
    label: str,
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

        return YOGO_CLASS_ORDERING.index(label_name)
    else:
        return YOGO_CLASS_ORDERING.index(label)


def load_labels(
    label_path: Path,
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
            # emtpy file, no labels, just keep moving
            return []

        if has_header:
            next(reader, None)

        for row in reader:
            assert (
                len(row) == 5
            ), f"should have [class,xc,yc,w,h] - got length {len(row)} {row}"

            label_idx = correct_label_idx(row[0], notes_data)

            # float for everything so we can make tensors of labels
            labels.append([float(label_idx)] + [float(v) for v in row[1:]])

    return labels


def label_file_to_tensor(
    label_path: Path,
    Sx: int,
    Sy: int,
    notes_data: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    "loads labels from label file into a tensor suitible for back prop, given by image path"

    try:
        labels = load_labels(label_path, notes_data=notes_data)
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
        normalize_images: bool = False,
        loader: Callable = read_grayscale,
        extensions: Optional[Tuple[str]] = ("png",),
        is_valid_file: Optional[Callable[[str], bool]] = None,
        *args,
        **kwargs,
    ):
        # the super().__init__ just sets transforms
        # the image_path is just for repr
        super().__init__(str(image_folder_path), *args, **kwargs)

        self.classes = YOGO_CLASS_ORDERING
        self.image_folder_path = image_folder_path
        self.label_folder_path = label_folder_path
        self.loader = loader
        self.normalize_images = normalize_images
        self.notes_data: Optional[Dict[str, Any]] = None

        # https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # essentially, to avoid dataloader workers from copying tonnes of mem,
        # we can't store samples in lists. Hence, the tensor and numpy array.
        image_paths, label_paths = self.make_dataset(
            Sx,
            Sy,
            is_valid_file=is_valid_file,
            extensions=extensions,
        )

        self.Sx = Sx
        self.Sy = Sy

        self._image_paths = np.array(image_paths).astype(np.string_)
        self._label_paths = np.array(label_paths).astype(np.string_)

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
        for label_file_path in self.label_folder_path.glob("*"):
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
            except StopIteration as e:
                # raise exception here? logic being that we want to know very quickly that we don't have
                # all the images we need. Open to changes, though.
                raise FileNotFoundError(
                    f"None of the following images exist: {possible_image_paths}"
                ) from e

            image_paths.append(str(image_file_path))
            label_paths.append(str(label_file_path))

        return image_paths, label_paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = str(self._image_paths[index], encoding="utf-8")
        label_path = str(self._label_paths[index], encoding="utf-8")
        image = self.loader(image_path)
        labels = label_file_to_tensor(
            Path(label_path), self.Sx, self.Sy, self.notes_data
        )
        if self.normalize_images:
            # turns our torch.uint8 tensor 'sample' into a torch.FloatTensor
            image = image / 255
        return image, labels

    def __len__(self) -> int:
        return len(self._image_paths)