#! /usr/bin/env python3

import sys

from pathlib import Path
from typing import List, Dict

from ruamel import yaml

from labelling_constants import CLASSES

"""
This file will scan through the labeled data and create the data set definition file.
Here is an example format!

```
class_names: ["healthy", "ring", "schizont", "troph"]
dataset_split_fractions:
  train: 0.70
  test:  0.25
  val:   0.05
dataset_paths:
  cellpose_sample:
    image_path: /tmp/training_data_sample_2022_11_01_cyto2/sample_2022_11_01
    label_path: /tmp/training_data_sample_2022_11_01_cyto2/sample_2022_11_01_labels_cyto2
```
"""


def class_names_from_classes_dot_txt(path_to_classes_dot_txt: Path) -> List[str]:
    with open(str(path_to_classes_dot_txt), "r") as f:
        return [s.strip() for s in f.readlines()]


def gen_labels(path_to_runset_folder: Path):
    folders = [Path(p).parent for p in path_to_runset_folder.glob("./**/images")]

    dataset_paths: Dict[str, Dict[str, str]] = dict()

    for i, folder_path in enumerate(folders):
        # check classes
        images_path = folder_path / "images"
        label_path = folder_path / "labels"

        if (not images_path.exists()) or (not label_path.exists()):
            print(
                f"WARNING: image path or label path doesn't exist: {images_path}, {label_path}. Continuing..."
            )
            continue

        dataset_paths[folder_path.name] = {
            "image_path": str(images_path),
            "label_path": str(label_path),
        }

    dataset_defs = {
        "class_names": CLASSES,
        "dataset_split_fractions": {"train": 0.75, "test": 0.20, "val": 0.05},
        "dataset_paths": dataset_paths,
    }

    yml = yaml.YAML()
    yml.indent(mapping=5, sequence=5, offset=3)

    with open("dataset_defs.yml", "w") as f:
        yml.dump(dataset_defs, f)
        print("dumped to dataset_defs.yml")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path to runset>")
        sys.exit(1)

    path_to_runset = Path(sys.argv[1])

    if not path_to_runset.exists():
        raise ValueError(f"{str(path_to_runset)} doesn't exist")

    gen_labels(path_to_runset)
