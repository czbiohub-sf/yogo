import csv
import json
import torch
import numpy as np

from pathlib import Path
from collections import defaultdict

import torchvision.ops as ops

from torchvision import datasets
from torchvision.io import read_image, ImageReadMode

from typing import List, Dict, Union, Tuple, Optional, Callable, Any, cast


LABEL_TENSOR_PRED_DIM_SIZE = 1 + 4 + 1
YOGO_CLASS_ORDERING = [
    "healthy",
    "ring",
    "trophozoite",
    "schizont",
    "gametocyte",
    "wbc",
    "misc",
]


def read_grayscale(img_path):
    try:
        return read_image(str(img_path), ImageReadMode.GRAY)
    except RuntimeError as e:
        raise RuntimeError(f"file {img_path} threw: {e}")


def split_labels_into_bins(
    labels: torch.Tensor, Sx, Sy
) -> Dict[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    labels shape is [N,5]; N is batch size, 5 is [label, x, y, x, y]
    """
    d: Dict[Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]] = defaultdict(list)
    for label in labels:
        i = torch.div((label[1] + label[3]) / 2, (1 / Sx), rounding_mode="trunc").long()
        j = torch.div((label[2] + label[4]) / 2, (1 / Sy), rounding_mode="trunc").long()
        d[(i, j)].append(label)
    return {k: torch.vstack(vs) for k, vs in d.items()}


def format_labels_tensor(labels: torch.Tensor, Sx: int, Sy: int) -> torch.Tensor:
    with torch.no_grad():
        output = torch.zeros(LABEL_TENSOR_PRED_DIM_SIZE, Sy, Sx)
        label_cells = split_labels_into_bins(labels, Sx, Sy)

        for (k, j), cell_label in label_cells.items():
            pred_square_idx = 0  # TODO this is a remnant of Sx,Sy being small; remove?
            output[0, j, k] = 1  # mask that there is a prediction here
            output[1:5, j, k] = cell_label[pred_square_idx][1:]  # xyxy
            output[5, j, k] = cell_label[pred_square_idx][0]  # prediction idx

        return output


def correct_label_idx(
    label: Union[str, int],
    dataset_classes: List[str],
    notes_data: Optional[Dict[str, Any]] = None,
) -> int:
    """
    dataset_classes is the ordering of classes that are given by
    label-studio. So we get the class of the prediction from
    `int(row[0])`, and get the index of that from YOGO_CLASS_ORDERING
    """
    if isinstance(label, int):
        return YOGO_CLASS_ORDERING.index(str(label))
    elif isinstance(label, str) and label.isnumeric():
        label = int(label)
        if notes_data is None:
            return YOGO_CLASS_ORDERING.index(dataset_classes[label])
        else:
            label_name: Optional[str] = None
            for row in notes_data["categories"]:
                if label == int(row["id"]):
                    label_name = row["name"]
                    break

            if label_name is None:
                raise ValueError(f"label index {label} not found in notes.json file")

            return YOGO_CLASS_ORDERING.index(label_name)
    raise ValueError(
        f"label must be an integer or a numeric string (i.e. '1', '2', ...); got {label}"
    )


def load_labels(
    label_path: Path,
    dataset_classes: List[str],
    notes_data: Optional[Dict[str, Any]] = None,
) -> List[List[float]]:
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
                return []

            if has_header:
                next(reader, None)

            for row in reader:
                assert (
                    len(row) == 5
                ), f"should have [class,xc,yc,w,h] - got length {len(row)} {row}"

                label_idx = correct_label_idx(row[0], dataset_classes, notes_data)

                # float for everything so we can make tensors of labels
                labels.append([float(label_idx)] + [float(v) for v in row[1:]])
    except FileNotFoundError:
        pass

    return labels


def label_file_to_tensor(
    label_path: Path,
    dataset_classes: List[str],
    Sx: int,
    Sy: int,
    notes_data: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    "loads labels from label file into a tensor suitible for back prop, given by image path"
    labels = load_labels(label_path, dataset_classes)
    labels_tensor = torch.Tensor(labels)

    if labels_tensor.nelement() == 0:
        return torch.zeros(LABEL_TENSOR_PRED_DIM_SIZE, Sy, Sx)

    labels_tensor[:, 1:] = ops.box_convert(labels_tensor[:, 1:], "cxcywh", "xyxy")
    return format_labels_tensor(labels_tensor, Sx, Sy)


class ObjectDetectionDataset(datasets.VisionDataset):
    def __init__(
        self,
        dataset_classes: List[str],
        image_path: Path,
        label_path: Path,
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
        super().__init__(str(image_path), *args, **kwargs)

        self.classes = YOGO_CLASS_ORDERING
        self.image_folder_path = image_path
        self.label_folder_path = label_path
        self.loader = loader
        self.normalize_images = normalize_images

        # https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # essentially, to avoid dataloader workers from copying tonnes of mem,
        # we can't store samples in lists. Hence, the tensor and numpy array.
        paths, tensors = self.make_dataset(
            Sx,
            Sy,
            is_valid_file=is_valid_file,
            extensions=extensions,
            dataset_classes=dataset_classes,
        )

        self._paths = np.array(paths).astype(np.string_)
        self._labels = torch.stack(tensors)

    def make_dataset(
        self,
        Sx: int,
        Sy: int,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        dataset_classes: List[str] = YOGO_CLASS_ORDERING,
    ) -> Tuple[List[str], List[torch.Tensor]]:
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

        notes_data = None
        if (self.label_folder_path.parent / "notes.json").exists():
            with open(str(self.label_folder_path.parent / "notes.json"), "r") as notes:
                notes_data = json.load(notes)

        # maps file name to a list of tuples of bounding boxes + classes
        paths: List[str] = []
        tensors: List[torch.Tensor] = []
        for label_file_path in self.label_folder_path.glob("*"):
            # ignore (*nix convention) hidden files
            if label_file_path.name.startswith("."):
                continue

            image_paths = [
                self.image_folder_path / label_file_path.with_suffix(sfx).name
                for sfx in [".png", ".jpg"]
            ]

            try:
                image_file_path = next(
                    ip for ip in image_paths if (ip.exists() and is_valid_file(str(ip)))
                )
            except StopIteration as e:
                # raise exception here? logic being that we want to know very quickly that we don't have
                # all the labels we need. Open to changes, though.
                raise FileNotFoundError(
                    f"None of the following images exist: {image_paths}"
                ) from e

            # if we have a `notes.json` file available, the labels are from a
            # label studio project, so use it. Otherwise, assume YOGO_CLASS_ORDERING

            labels = label_file_to_tensor(
                label_file_path, dataset_classes, Sx, Sy, notes_data
            )
            paths.append(str(image_file_path))
            tensors.append(labels)

        return paths, tensors

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = str(self._paths[index], encoding="utf-8")
        sample = self.loader(img_path)
        target = self._labels[index, ...]
        if self.normalize_images:
            # turns our torch.uint8 tensor 'sample' into a torch.FloatTensor
            sample = sample / 256
        return sample, target

    def __len__(self) -> int:
        return len(self._paths)
