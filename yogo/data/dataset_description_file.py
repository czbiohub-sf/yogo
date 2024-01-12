import warnings

from ruamel.yaml import YAML
from pathlib import Path
from dataclasses import dataclass

from typing import Any, List, Dict, Optional, Literal

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

Here are many examples: https://github.com/czbiohub-sf/lfm-dataset-definitions/

Required Fields
---------------

A dataset definition file is a YAML file with a `dataset_paths` key, with a list of dataset
path specifications as values. Dataset specifications are another key-value pair, where
the key is an arbitrary label for humans - it is not used by the parsing code. The value can
be either (a) `defn_path` which points to another definition file to be loaded (a "Literal
Specification"), or (b) an `image_path` and a `label_path` pair (a "Recursive Specification").
All paths are absolute. Here's an example

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
restrictions. The dataset definition specifcation is a graph, where the nodes are Dataset
Definitions. Edges are directed, and are from the Definition to the Definitions that it defines.
For practical reasons, we can't accept arbitrary graph definitions. For example, if the
specification has a cycle, we will have to reject it (only trees are allowed). We'll also
choose to use unique paths - that is, for any Dataset Definition in our tree, there exists
only one path to it. This'll make it easier keep track of folders. Stricter == Better. Essentially,
we're defining a Tree ( https://en.wikipedia.org/wiki/Tree_(graph_theory) ).

Further required fields for the "Top Level" definition file are:
    - classes: a list of class names to be used in the dataset. Conflicting class definitions will be rejected.
    - split_fractions: a dictionary specifying the split fractions for the dataset. Keys can be
    `train`, `val`, and `test`. If `test_paths` is present, `train` should be left out. The values
    are floats between 0 and 1, and the sum of `split_fractions` should be 1. May be depricated!

Optional Fields
---------------

Optional fields include:
    - test_paths: similar to dataset_paths, but for the test set. Basically, it's just a way
    to explicitly specify which data is isolated for testing.
    - thumbnail_augmentation: a dictionary specifying a class name and pointing to a directory
    of thumbnails. Somewhat niche. Ideally we'd have some sort of other "arbitrary metadata"
    specification that could be used for this sort of thing.
"""


class SplitFractions:
    def __init__(
        self, train: Optional[float], val: Optional[float], test: Optional[float]
    ) -> None:
        self.train = train or 0
        self.val = val or 0
        self.test = test or 0

        if not (self.train + self.val + self.test - 1) < 1e-10:
            raise ValueError(
                f"train, val, and test must sum to 1; they sum to {self.train + self.val + self.test}"
            )

    @classmethod
    def from_dict(
        cls, dct: Dict[str, float], test_paths_present: bool = True
    ) -> "SplitFractions":
        if test_paths_present and "train" in dct:
            raise InvalidDatasetDefinitionFile(
                "when `test_paths` is present in a dataset descriptor file, 'train' "
                "is not a valid key for `dataset_split_fractions`, since we will use "
                "all the data from `dataset_paths` for training"
            )
        if not any(v in dct for v in ("train", "val", "test")):
            raise InvalidDatasetDefinitionFile(
                f"dct must have keys `train`, `val`, and `test` - found keys {dct.keys()}"
            )
        if len(dct) > 3:
            raise InvalidDatasetDefinitionFile(
                f"dct must have keys `train`, `val`, and `test` only, but found {len(dct)} keys"
            )
        return cls(dct.get("train", None), dct.get("val", None), dct.get("test", None))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SplitFractions):
            return False
        return (
            self.train == other.train
            and self.val == other.val
            and self.test == other.test
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            **({"train": self.train} if self.train is not None else {}),
            **({"val": self.val} if self.val is not None else {}),
            **({"test": self.test} if self.test is not None else {}),
        }

    def keys(self) -> List[str]:
        return list(self.to_dict().keys())

    def partition_sizes(self, total_size: int) -> Dict[str, int]:
        split_fractions = self.to_dict()
        keys = self.keys()

        dataset_sizes = {k: round(split_fractions[k] * total_size) for k in keys[:-1]}
        final_dataset_size = {keys[-1]: total_size - sum(dataset_sizes.values())}
        split_sizes = {**dataset_sizes, **final_dataset_size}

        all_sizes_are_gt_0 = all([sz >= 0 for sz in split_sizes.values()])
        split_sizes_eq_dataset_size = sum(split_sizes.values()) == total_size

        if not (all_sizes_are_gt_0 and split_sizes_eq_dataset_size):
            raise ValueError(
                f"could not create valid dataset split sizes: {split_sizes}, "
                f"full dataset size is {total_size}"
            )

        return split_sizes


class InvalidDatasetDefinitionFile(Exception):
    ...


@dataclass
class LiteralSpecification:
    """
    This defines an (image dir path, label dir path) pair. In the
    specification above, it's a "Literal Specification".

    Mostly, it just gives us a lot of convenience. Defining __hash__
    and __eq__ make it easy to check for duplicates in a list. from_dict
    and to_dict make it easy to serialize and deserialize from the raw
    yaml.
    """

    image_path: Path
    label_path: Path

    @classmethod
    def from_dict(self, dct: Dict[str, str]) -> "LiteralSpecification":
        if len(dct) != 2:
            raise InvalidDatasetDefinitionFile(
                f"LiteralSpecification must have two keys; found {len(dct)}"
            )
        elif "image_path" not in dct or "label_path" not in dct:
            defn_path_hint = (
                (
                    " 'defn_path' found - this is a coding error, "
                    "please fill out a new issue here: "
                    "https://github.com/czbiohub-sf/yogo/issues/new"
                )
                if "defn_path" in dct
                else ""
            )

            raise InvalidDatasetDefinitionFile(
                "LiteralSpecification must have keys 'image_path' and 'label_path'"
                + defn_path_hint
            )
        else:
            return LiteralSpecification(
                Path(dct["image_path"]), Path(dct["label_path"])
            )

    def to_dict(self) -> Dict[str, str]:
        return {"image_path": str(self.image_path), "label_path": str(self.image_path)}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LiteralSpecification):
            return False
        else:
            return (
                self.image_path == other.image_path
                and self.label_path == other.label_path
            )

    def __hash__(self) -> int:
        return hash((self.image_path, self.label_path))


@dataclass
class DatasetDefinition:
    """The actual Dataset Definition!

    Although it's representation on disc (via the yml files) is recursive,
    this is not. When we load the yml files, we flatten the structure and
    check for cycles / duplicates.

    The main results are mostly the same as before:
        - dataset_paths: a list of dicts, {image_path: str, label_path: str}
        - test_dataset_paths: a list of dicts, {image_path: str, label_path: str}
        - classes: a list of class names
        - thumbnail_augmentation: a dict of {class_name: Path}
    """

    _dataset_paths: List[LiteralSpecification]
    _test_dataset_paths: List[LiteralSpecification]

    classes: Optional[List[str]]
    thumbnail_augmentation: Optional[Dict[str, Path]]
    split_fractions: SplitFractions

    @property
    def dataset_paths(self) -> List[Dict[str, str]]:
        return [dp.to_dict() for dp in self._dataset_paths]

    @property
    def test_dataset_paths(self) -> List[Dict[str, str]]:
        return [dp.to_dict() for dp in self._test_dataset_paths]

    @classmethod
    def from_yaml(cls, path: Path) -> "DatasetDefinition":
        """
        The general idea here is that `dataset_paths` has a list of
        dataset specifications, which can be literal or recursive. We'll
        make a list of both, and then try to to_dict the recursive specifications.
        We to_dict the recursive specifications later so we can
        """
        with open(path, "r") as f:
            yaml = YAML(typ="safe")
            data = yaml.load(f)

        if "dataset_paths" not in data:
            raise InvalidDatasetDefinitionFile(
                f"Missing dataset_paths for definition file at {path}"
            )

        dataset_specs = cls._load_dataset_specifications(data["dataset_paths"].values())

        if "test_paths" in data:
            test_specs = cls._load_dataset_specifications(
                data["test_paths"].values(),
                exclude_ymls=[path],
                exclude_specs=dataset_specs,
                dataset_paths_key="test_paths",
            )
            test_paths_present = True
        else:
            test_specs = []
            test_paths_present = False

        classes = data.get("classes", None)

        return DatasetDefinition(
            _dataset_paths=dataset_specs,
            _test_dataset_paths=test_specs,
            classes=classes,
            thumbnail_augmentation=cls.load_thumbnails(classes, data),
            split_fractions=SplitFractions.from_dict(
                data["dataset_split_fractions"], test_paths_present=test_paths_present
            ),
        )

    @staticmethod
    def _load_dataset_specifications(
        specs: List[Dict[str, str]],
        exclude_ymls: List[Path] = [],
        exclude_specs: List[LiteralSpecification] = [],
        dataset_paths_key: Literal["test_paths", "dataset_paths"] = "dataset_paths",
    ) -> List[LiteralSpecification]:
        """
        load the list of dataset specifications into a list
        of LiteralSpecification. Essentially, we try to to_dict
        any recursive specifications into literal specifications.

        >>> extract_paths = _extract_dataset_paths(yml_path)

        We also do some checking here for cycles (as defined by
        `exclude_ymls`) or duplicates.

        `exclude_specs` is a list of specifications that
        should be excluded for one reason or another. for example,
        if a literal specifcation is in the training set, you want
        to make sure you exclude it in the testing set.
        """
        literal_defns: List[LiteralSpecification] = []

        for spec in specs:
            if "defn_path" in spec:
                # extract the paths recursively!
                new_yml_file = Path(spec["defn_path"])

                if new_yml_file in exclude_ymls:
                    raise InvalidDatasetDefinitionFile(
                        f"cycle found: {spec['defn_path']} is duplicated"
                    )

                child_specs = DatasetDefinition._load_dataset_specifications(
                    _extract_dataset_paths(Path(new_yml_file)),
                    exclude_ymls=[new_yml_file, *exclude_ymls],
                    dataset_paths_key=dataset_paths_key,
                )

                literal_defns.extend(child_specs)
            elif "image_path" in spec and "label_path" in spec:
                # ez case
                literal_defns.append(LiteralSpecification.from_dict(spec))
            else:
                # even easier case
                raise InvalidDatasetDefinitionFile(
                    f"Invalid spec in dataset_paths: {spec}"
                )

        # check that all of our paths are unique
        if len(set(literal_defns + exclude_specs)) != len(
            literal_defns + exclude_specs
        ):
            # duplicate literal definitions, or one of the literal definitions that we found
            # is in the exclude set. Report them!
            # TODO report which ones are bad <|:^|
            raise InvalidDatasetDefinitionFile(
                "literal definition found in exclude paths!"
            )

        return literal_defns

    @staticmethod
    def load_thumbnails(
        classes: List[str], yaml_data: Dict[str, Any]
    ) -> Optional[Dict[str, Path]]:
        if "thumbnail_agumentation" in yaml_data:
            class_to_thumbnails = yaml_data["thumbnail_agumentation"]
            if not isinstance(class_to_thumbnails, dict):
                raise InvalidDatasetDefinitionFile(
                    "thumbnail_agumentation must map class names to paths to thumbnail "
                    "directories (e.g. `misc: /path/to/thumbnails/misc`)"
                )

            for k in class_to_thumbnails:
                if k not in classes:
                    raise InvalidDatasetDefinitionFile(
                        f"thumbnail_agumentation class {k} is not a valid class name"
                    )
            return class_to_thumbnails
        return None


def check_dataset_paths(
    dataset_paths: List[LiteralSpecification], prune: bool = False
) -> None:
    to_prune: List[int] = []
    for i in range(len(dataset_paths)):
        if not (
            dataset_paths[i].image_path.is_dir()
            and dataset_paths[i].label_path.is_dir()
            and len(list(dataset_paths[i].label_path.iterdir())) > 0
        ):
            if prune:
                warnings.warn(
                    f"image_path or label_path do not lead to a directory, or there are no labels.\n"
                    f"image_path={dataset_paths[i].image_path}\n"
                    f"label_path={dataset_paths[i].label_path}\n"
                    f"will prune."
                )
                to_prune.append(i)
            else:
                raise FileNotFoundError(
                    f"image_path or label_path do not lead to a directory, or there are no labels.\n"
                    f"image_path={dataset_paths[i].image_path}\n"
                    f"label_path={dataset_paths[i].label_path}"
                )

    # reverse order so we don't move around the to-delete items in the list
    for i in to_prune[::-1]:
        del dataset_paths[i]


def _extract_dataset_paths(path: Path) -> List[Dict[str, str]]:
    """
    convert List[Dict[str,Dict[str,str]]] to List[Dict[str,str]],
    since the enclosing dict has only 1 kv pair and 1 value
    """
    with open(path, "r") as f:
        yaml = YAML(typ="safe")
        data = yaml.load(f)

    if "dataset_paths" not in data:
        raise InvalidDatasetDefinitionFile(
            f"Missing dataset_paths for definition file at {path}"
        )

    return list(data["dataset_paths"].values())
