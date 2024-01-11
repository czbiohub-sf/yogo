import warnings

from ruamel import yaml
from pathlib import Path
from dataclasses import dataclass

from typing import Any, List, Dict, Optional, Iterator

from yogo.data import YOGO_CLASS_ORDERING

""" I don't *love* the dataset definition that's been defined here anymore.

Here are the issues:
    - 'split fractions' are a part of the definition file, meaning that if we
    want to change the split, we need to change the definition file. This is
    occuring more than expected, so I think it should be separated
    - it's hard to keep track of all the combinations of dataset paths. Looking
    into it this morning, I've found some discrepancies (specifically some
    dataset paths that are missing from a given definition file) that should be
    fixed
    - it's hard to find these discrepancies, since it's just a list of paths. When
    you have > 10, it's difficult for the human brain to find duplicates.
    - `load_dataset_description` is a horrendously long and difficult-to-read
    function. This should be easy to parse!

Things that are good:
    - the 'definition file' method of collecting data is great for modularity and
    organization. I've found myself using this file a lot outside of YOGO, which
    lends credability to it's usefulness.

Potential improvements:
    - recursive definitions: I should be able to list some specific paths in a file
    and reference that file in another. E.g. a "uganda-healthy" dataset definition
    could be imported into a "uganda" dataset definition and another file. This would
    simplify the composition of these files tremendously.
    - test tools: should be able to have a tool to check the validity of dataset
    definitions, such as looking for duplicate paths (perhaps in this file, in an
    'if __main__' block, or maybe in the YOGO cli?)
    - moving split-fractions to YOGO: I'm somewhat undecided. This is a more minor fix
    compared to the above.

------------------------------------------------------------------------------------------------------------------

New Specification
-----------------

Required Fields
-------------

A dataset definition file is a YAML file with a `dataset_paths` key, with a list of dataset
path specifications as values. Dataset specifications are another key-value pair, where
the key is an arbitrary label for humans - it is not used by the parsing code. The value can
be either (a) `defn_path` which points to another definition file to be loaded (a "Literal
Specification"), or (b) an `image_path` and a `label_path` pair (a "Recursive Specification").
Here's an example

```yaml
dataset_paths:
    image_and_label_dirs:               # These three lines make up one Dataset Specification
        image_path: /path/to/images     # This Dataset Specification is a "Literal Specification"
        label_path: /path/to/labels     # since it defines the actual image and label paths
    another dataset_defn:                                # These two lines make up another Dataset Specification.
        defn_path: /path/to/another/dataset_defn.yml     # This Dataset Specification is a "Recursive Specification".

# the composition of each of the Dataset Specifications above gives a full Dataset Definition.
```

Note: the ability to specify another dataset definition within a dataset definition has some
restrictions. The dataset definition specifcation is a graph. For practical reasons, we can't
accept arbitrary graph definitions. For example, if the specification has a cycle, we will have
to reject it (only trees are allowed).

Optional Fields
---------------

Optional fields include:
    - classes: a list of class names to be used in the dataset. Conflicting class definitions
    will be rejected.
    - test_paths: similar to dataset_paths, but for the test set. Basically, it's just a way
    to explicitly specify which data is isolated for testing.
    - split_fractions: a dictionary specifying the split fractions for the dataset. Keys can be
    `train`, `val`, and `test`. If `test_paths` is preset, `train` should be left out. The values
    are floats between 0 and 1, and the sum of `split_fractions` should be 1. WILL BE DEPRICATED SOON.
    - thumbnail_augmentation: a dictionary specifying a class name and pointing to a directory
    of thumbnails. Somewhat niche. Ideally we'd have some sort of other "arbitrary metadata"
    specification that could be used for this sort of thing.
"""


class InvalidDatasetDescriptionFile(Exception):
    ...


@dataclass
class DatasetDescription:
    classes: Optional[List[str]]
    split_fractions: Dict[str, float]
    dataset_paths: List[Dict[str, Path]]
    test_dataset_paths: Optional[List[Dict[str, Path]]]
    thumbnail_augmentation: Optional[Dict[int, Path]]

    # yuck! Iterator over many different types.
    # Used to easily split out vars from this
    # dataclass, but it's probably better to
    # explicitly index the DatasetDescription
    # object
    def __iter__(self) -> Iterator[Any]:
        return iter(
            (
                self.classes,
                self.split_fractions,
                self.dataset_paths,
                self.test_dataset_paths,
                self.thumbnail_augmentation,
            )
        )


def check_dataset_paths(
    dataset_paths: List[Dict[str, Path]], prune: bool = False
) -> None:
    to_prune: List[int] = []
    for i in range(len(dataset_paths)):
        if not (
            dataset_paths[i]["image_path"].is_dir()
            and dataset_paths[i]["label_path"].is_dir()
            and len(list(dataset_paths[i]["label_path"].iterdir())) > 0
        ):
            if prune:
                warnings.warn(
                    f"image_path or label_path do not lead to a directory, or there are no labels\n"
                    f"image_path={dataset_paths[i]['image_path']}\nlabel_path={dataset_paths[i]['label_path']}"
                )
                to_prune.append(i)
            else:
                raise FileNotFoundError(
                    f"image_path or label_path do not lead to a directory\n"
                    f"image_path={dataset_paths[i]['image_path']}\nlabel_path={dataset_paths[i]['label_path']}"
                )

    # reverse order so we don't move around the to-delete items in the list
    for i in to_prune[::-1]:
        del dataset_paths[i]


def load_dataset_description(dataset_description: str) -> DatasetDescription:
    """Loads and validates dataset description file"""
    required_keys = [
        "class_names",
        "dataset_split_fractions",
        "dataset_paths",
    ]
    with open(dataset_description, "r") as desc:
        yaml_data = yaml.safe_load(desc)

        for k in required_keys:
            if k not in yaml_data:
                raise InvalidDatasetDescriptionFile(
                    f"{k} is required in dataset description files, but was "
                    f"found missing for {dataset_description}"
                )

        classes = yaml_data["class_names"]
        split_fractions = {
            k: float(v) for k, v in yaml_data["dataset_split_fractions"].items()
        }
        dataset_paths = [
            {k: Path(v) for k, v in d.items()}
            for d in yaml_data["dataset_paths"].values()
        ]
        check_dataset_paths(dataset_paths, prune=True)

        if "test_paths" in yaml_data:
            test_dataset_paths = [
                {k: Path(v) for k, v in d.items()}
                for d in yaml_data["test_paths"].values()
            ]
            check_dataset_paths(test_dataset_paths, prune=False)

            # when we have 'test_paths', all the data from dataset_paths
            # will be used for training, so we should only have 'test' and
            # 'val' in dataset_split_fractions.
            if "val" not in split_fractions or "test" not in split_fractions:
                raise InvalidDatasetDescriptionFile(
                    "'val' and 'test' are required keys for dataset_split_fractions"
                )
            if "train" in split_fractions:
                raise InvalidDatasetDescriptionFile(
                    "when `test_paths` is present in a dataset descriptor file, 'train' "
                    "is not a valid key for `dataset_split_fractions`, since we will use "
                    "all the data from `dataset_paths` for training"
                )
        else:
            test_dataset_paths = None
            if any(k not in split_fractions for k in ("test", "train", "val")):
                raise InvalidDatasetDescriptionFile(
                    "'train', 'val', and 'test' are required keys for dataset_split_fractions - missing at least one. "
                    f"split fractions was {split_fractions}"
                )

        thumbnail_data: Optional[Dict[int, Path]]
        if "thumbnail_agumentation" in yaml_data:
            class_to_thumbnails = yaml_data["thumbnail_agumentation"]
            if not isinstance(class_to_thumbnails, dict):
                raise InvalidDatasetDescriptionFile(
                    "thumbnail_agumentation must map class names to paths to thumbnail "
                    "directories (e.g. `misc: /path/to/thumbnails/misc`)"
                )

            thumbnail_data = dict()
            for k in class_to_thumbnails:
                if k not in YOGO_CLASS_ORDERING:
                    raise InvalidDatasetDescriptionFile(
                        f"thumbnail_agumentation class {k} is not a valid class name"
                    )
                thumbnail_data[YOGO_CLASS_ORDERING.index(k)] = Path(
                    class_to_thumbnails[k]
                )
        else:
            thumbnail_data = None

        if not sum(split_fractions.values()) == 1:
            raise InvalidDatasetDescriptionFile(
                "invalid split fractions for dataset: split fractions must add to 1, "
                f"got {split_fractions}"
            )

        return DatasetDescription(
            classes, split_fractions, dataset_paths, test_dataset_paths, thumbnail_data
        )
