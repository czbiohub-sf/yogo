from ruamel import yaml
from pathlib import Path
from dataclasses import dataclass

from typing import List, Dict, Optional

from yogo.data.dataset import YOGO_CLASS_ORDERING


class InvalidDatasetDescriptionFile(Exception):
    ...


@dataclass
class DatasetDescription:
    classes: Optional[List[str]]
    split_fractions: Dict[str, float]
    dataset_paths: List[Dict[str, Path]]
    test_dataset_paths: Optional[List[Dict[str, Path]]]
    thumbnail_augmentation: Optional[Dict[int, Path]]

    def __iter__(self):
        return iter(
            (
                self.classes,
                self.split_fractions,
                self.dataset_paths,
                self.test_dataset_paths,
                self.thumbnail_augmentation,
            )
        )


def check_dataset_paths(dataset_paths: List[Dict[str, Path]], prune: bool = False):
    to_prune: List[int] = []
    for i in range(len(dataset_paths)):
        if not (
            dataset_paths[i]["image_path"].is_dir()
            and dataset_paths[i]["label_path"].is_dir()
            and len(list(dataset_paths[i]["label_path"].iterdir())) > 0
        ):
            if prune:
                print(f"pruning {dataset_paths[i]}")
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
        "class_names" "dataset_split_fractions",
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
