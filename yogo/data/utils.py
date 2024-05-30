import os
import torch
import warnings

from time import sleep
from pathlib import Path
from typing import Union, Optional, Tuple, List
from ruamel.yaml import YAML

from torchvision.io import read_image as read_image_torch, ImageReadMode

from yogo.data.data_transforms import MultiArgSequential
from yogo.data.dataset_definition_file import DatasetDefinition


def read_image(img_path: Union[str, Path], rgb: bool = False) -> torch.Tensor:
    img_mode = ImageReadMode.RGB if rgb else ImageReadMode.GRAY
    try:
        return read_image_torch(str(img_path), img_mode)
    except RuntimeError as e:
        raise RuntimeError(f"file {img_path} threw: {e}") from e


def read_image_robust(
    img_path: Union[str, Path],
    retries: int = 3,
    min_duration: float = 0.1,
    rgb: bool = False,
) -> Optional[torch.Tensor]:
    """
    Attempts to read an image file with retry logic.

    This function tries to read an image with a specified number of retries. If all attempts fail,
    it logs a warning and returns None.
    """
    img_mode = ImageReadMode.RGB if rgb else ImageReadMode.GRAY
    for i in range(retries):
        try:
            return read_image_torch(str(img_path), img_mode)
        except Exception as e:
            warnings.warn(f"file {img_path} threw: {e}")
            if i == retries - 1:
                warnings.warn(f"all attempts to read {img_path} failed")
                break
            sleep(min_duration * (2**retries))
    return None


def collate_batch_robust(
    batch: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
    transforms: MultiArgSequential = MultiArgSequential(),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for batching data, handling optional items robustly.

    Filters out None items from a batch and applies transformations to the batched inputs and labels. This function
    is designed to work with datasets that may return None for some items, allowing the DataLoader to skip these items
    gracefully.
    """
    inputs, labels = zip(*[pair for pair in batch if pair is not None])
    batched_inputs = torch.stack(inputs)
    batched_labels = torch.stack(labels)
    return transforms(batched_inputs, batched_labels)


def convert_dataset_definition_to_ultralytics_format(
    dataset_definition_path: Path, target_dir: Path
):
    """
    Ultralytics has a difficult and restrictive format for ingesting data for
    training. This function converts our definition file to theirs. Note that we create symlinks
    to our data in the target directory.

    note: this function is somewhat hacky currently, and is not expected to be used
    frequently.

    reference:
    https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#22-create-labels
    """
    dataset_definition = DatasetDefinition.from_yaml(dataset_definition_path)

    classes = dataset_definition.classes

    target_dir.mkdir(exist_ok=True, parents=True)

    train_dir = target_dir / "train"
    val_dir = target_dir / "val"

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    train_dir_paths = []
    for spec in dataset_definition.dataset_paths:
        (train_dir / spec.image_path.parent.name).mkdir(exist_ok=True)

        try:
            os.symlink(
                spec.image_path, train_dir / spec.image_path.parent.name / "images"
            )
        except FileExistsError:
            pass
        try:
            os.symlink(
                spec.label_path, train_dir / spec.image_path.parent.name / "labels"
            )
        except FileExistsError:
            pass

        train_dir_paths.append(str(train_dir / spec.image_path.parent.name / "images"))

    test_dir_paths = []
    for spec in dataset_definition.test_dataset_paths:
        (val_dir / spec.image_path.parent.name).mkdir(exist_ok=True)

        try:
            os.symlink(
                spec.image_path, val_dir / spec.image_path.parent.name / "images"
            )
        except FileExistsError:
            pass
        try:
            os.symlink(
                spec.label_path, val_dir / spec.image_path.parent.name / "labels"
            )
        except FileExistsError:
            pass

        test_dir_paths.append(str(val_dir / spec.image_path.parent.name / "images"))

    yaml = YAML()
    yaml.default_flow_style = False
    ultralytics_defn = {
        "path": str(target_dir.resolve()),
        "train": train_dir_paths,
        "val": test_dir_paths,
        "names": dict(enumerate(classes)),
    }
    yaml.dump(ultralytics_defn, target_dir / "dataset_defn.yaml")
