import os
import torch

from typing import Union
from pathlib import Path
from ruamel.yaml import YAML

from torchvision.io import read_image, ImageReadMode

from yogo.data import YOGO_CLASS_ORDERING
from yogo.data.dataset_definition_file import DatasetDefinition


def read_grayscale(img_path: Union[str, Path]) -> torch.Tensor:
    try:
        return read_image(str(img_path), ImageReadMode.GRAY)
    except RuntimeError as e:
        raise RuntimeError(f"file {img_path} threw: {e}")


def convert_dataset_definition_to_ultralytics_format(
    dataset_definition_path: Path, target_dir: Path
):
    """
    Ultralytics has a difficult and restrictive format for ingesting data for
    training. This function converts our definition file to theirs. Note that we create symlinks
    to our data in the target directory.

    note: this function is somewhat hacky currently, and is not expected to be used
    frequently. Ask Axel and he'll help run it.
    """
    dataset_definition = DatasetDefinition.from_yaml(dataset_definition_path)

    classes = dataset_definition.classes or YOGO_CLASS_ORDERING

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
