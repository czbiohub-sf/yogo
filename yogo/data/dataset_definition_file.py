import warnings

from enum import Enum
from pathlib import Path
from ruamel.yaml import YAML
from dataclasses import dataclass
from typing import Any, Set, List, Dict, Tuple, Union, Optional

from yogo.data.split_fractions import SplitFractions

"""
Specification
-----------------

See in-depth docs here: https://github.com/czbiohub-sf/yogo/blob/main/docs/dataset-definition.md

Required Fields
---------------

A dataset definition file is a YAML file with a `dataset_paths` key, with a list of dataset
path specifications as values. Dataset specifications are another key-value pair, where
the key is an arbitrary label for humans - it is not used by the parsing code. The value can
be either (a) `defn_path` which points to another definition file to be loaded (a "Recursive
Specification"), or (b) an `image_path` and a `label_path` pair (a "Literal Specification").
All paths are absolute. Here's an example

```yaml
# < other required fields >
dataset_paths:
    image_and_label_dirs:               # These three lines make up one Dataset Specification
        image_path: /path/to/images     # This Dataset Specification is a "Literal Specification"
        label_path: /path/to/labels     # since it defines the actual image and label paths
    another_dataset_defn:                                # These two lines make up another Dataset Specification.
        defn_path: /path/to/another/dataset_defn.yml     # This Dataset Specification is a "Recursive Specification".

# the composition of each of the Dataset Specifications above gives a full Dataset Definition.
```

Other Required Fields:

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


Recursive Specifications
------------------------

They can be either relative or absolute. If they are relative, they're relative to the
parent directory of the definition file. So, if the definition file is in /path/to/defn_file.yml,
and was

```yaml
# < other required fields >
dataset_paths:
    in_this_dir:
        defn_path: dataset_defn.yml
    in_another_dir:
        defn_path: ../cool-data/dataset_defn.yml
```

then the folder structure would be

```console
$ tree /path
.
├── cool-data
│   └── defn_file.yml
└── to
    └── defn_file.yml

```

Note: the ability to specify another dataset definition within a dataset definition has some
restrictions. The dataset definition specifcation is a graph, where the nodes are Dataset
Definitions. Edges are directed, and are from the Definition to the Definitions that it defines.
For practical reasons, we can't accept arbitrary graph definitions. For example, if the
specification has a cycle, we will have to reject it (only trees are allowed). We'll also
choose to use unique paths - that is, for any Dataset Definition in our tree, there exists
only one path to it. This'll make it easier keep track of folders. Stricter == Better. Essentially,
we're defining a Tree ( https://en.wikipedia.org/wiki/Tree_(graph_theory) ).
"""


class InvalidDatasetDefinitionFile(Exception): ...


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
    def from_dict(cls, dct: Dict[str, str]) -> "LiteralSpecification":
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
        return {"image_path": str(self.image_path), "label_path": str(self.label_path)}

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


class SpecificationsKey(Enum):
    DATASET_PATHS = "dataset_paths"
    TEST_DATASET_PATHS = "test_paths"
    ALL_DATASET_PATHS = "all_paths"


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

    _dataset_paths: Set[LiteralSpecification]
    _test_dataset_paths: Set[LiteralSpecification]

    classes: List[str]
    thumbnail_augmentation: Optional[Dict[str, Union[Path, List[Path]]]]
    split_fractions: SplitFractions

    @property
    def dataset_paths(self) -> List[LiteralSpecification]:
        return list(self._dataset_paths)

    @property
    def test_dataset_paths(self) -> List[LiteralSpecification]:
        return list(self._test_dataset_paths)

    @property
    def all_dataset_paths(self) -> List[LiteralSpecification]:
        return list(self._dataset_paths | self._test_dataset_paths)

    @classmethod
    def from_yaml(cls, path: Path) -> "DatasetDefinition":
        """
        Load ddf file from the yaml file definition.
        """
        path = Path(path)  # defensive, in-case we're handed a string

        with open(path, "r") as f:
            yaml = YAML(typ="safe")
            data = yaml.load(f)

        test_paths_present = "test_paths" in data

        try:
            classes = data["class_names"]
        except KeyError as e:
            raise InvalidDatasetDefinitionFile(
                "`classes` is a required key in the dataset definition file"
            ) from e

        if test_paths_present:
            dataset_specs = cls._load_dataset_specifications(
                path, classes, dataset_paths_key=SpecificationsKey.DATASET_PATHS
            )
            test_specs = cls._load_dataset_specifications(
                path,
                classes,
                exclude_ymls=[path],
                exclude_specs=dataset_specs,
                dataset_paths_key=SpecificationsKey.TEST_DATASET_PATHS,
            )
        else:
            dataset_specs = cls._load_dataset_specifications(
                path, classes, dataset_paths_key=SpecificationsKey.ALL_DATASET_PATHS
            )
            test_specs = set()

        dataset_specs = DatasetDefinition._check_dataset_paths(dataset_specs)
        test_specs = DatasetDefinition._check_dataset_paths(test_specs)
        if "dataset_split_fractions" in data:
            split_fractions = SplitFractions.from_dict(
                data["dataset_split_fractions"], test_paths_present=test_paths_present
            )
        else:
            split_fractions = SplitFractions.train_only()

        return cls(
            _dataset_paths=dataset_specs,
            _test_dataset_paths=test_specs,
            classes=classes,
            thumbnail_augmentation=DatasetDefinition._load_thumbnails(classes, data),
            split_fractions=split_fractions,
        )

    def __add__(self, other: "DatasetDefinition") -> "DatasetDefinition":
        """
        return a new dataset definition that's the concatenation of this definition
        and another. The classes, thumbnail augmentation, and split fractions must
        match.
        """
        if self.classes != other.classes:
            raise ValueError(
                "cannot concatenate two dataset definitions with different classes"
            )
        elif self.thumbnail_augmentation != other.thumbnail_augmentation:
            raise ValueError(
                "cannot concatenate two dataset definitions with different thumbnail augmentation"
            )
        elif self.split_fractions != other.split_fractions:
            raise ValueError(
                "cannot concatenate two dataset definitions with different split fractions"
            )

        _dataset_paths = self._dataset_paths | other._dataset_paths
        _test_dataset_paths = self._test_dataset_paths | other._test_dataset_paths

        return DatasetDefinition(
            _dataset_paths=_dataset_paths,
            _test_dataset_paths=_test_dataset_paths,
            classes=self.classes,
            thumbnail_augmentation=self.thumbnail_augmentation,
            split_fractions=self.split_fractions,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DatasetDefinition):
            return False
        return (
            self._dataset_paths == other._dataset_paths
            and self._test_dataset_paths == other._test_dataset_paths
            and self.classes == other.classes
            and self.thumbnail_augmentation == other.thumbnail_augmentation
            and self.split_fractions == other.split_fractions
        )

    @staticmethod
    def _extract_specs(
        yml_path: Path, dataset_paths_key: SpecificationsKey
    ) -> Tuple[List[str], List[Dict[str, str]]]:
        with open(yml_path, "r") as f:
            yaml = YAML(typ="safe")
            data = yaml.load(f)

        try:
            classes = data["class_names"]
        except KeyError:
            raise InvalidDatasetDefinitionFile(
                "`classes` is a required key in the dataset definition file"
            )

        if dataset_paths_key == SpecificationsKey.ALL_DATASET_PATHS:
            dataset_paths = list(
                data.get(SpecificationsKey.DATASET_PATHS.value, dict()).values()
            )
            test_paths = list(
                data.get(SpecificationsKey.TEST_DATASET_PATHS.value, dict()).values()
            )
            specs = dataset_paths + test_paths
        elif dataset_paths_key.value not in data:
            # catch case where there are no test_paths but dataset_paths_key is TEST_DATASET_PATHS
            specs = []
        else:
            specs = data[dataset_paths_key.value].values()

        return classes, specs

    @staticmethod
    def _load_dataset_specifications(
        yml_path: Path,
        classes: List[str],
        exclude_ymls: List[Path] = [],
        exclude_specs: Set[LiteralSpecification] = set(),
        dataset_paths_key: SpecificationsKey = SpecificationsKey.DATASET_PATHS,
    ) -> Set[LiteralSpecification]:
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
        literal_defns: Set[LiteralSpecification] = set()

        spec_classes, specs = DatasetDefinition._extract_specs(
            yml_path, dataset_paths_key
        )

        if spec_classes != classes:
            raise InvalidDatasetDefinitionFile(f"classes mismatch in {yml_path}")

        for spec in specs:
            if "defn_path" in spec:
                # here we extract the paths recursively!

                # resolve paths relative to the current yml path
                new_yml_path = Path(spec["defn_path"])
                if not new_yml_path.is_absolute():
                    new_yml_path = yml_path.parent / new_yml_path

                # check for cycles
                if new_yml_path in exclude_ymls:
                    raise InvalidDatasetDefinitionFile(
                        f"cycle found: {spec['defn_path']} is duplicated"
                    )

                # recur!
                child_specs = DatasetDefinition._load_dataset_specifications(
                    new_yml_path,
                    classes,
                    exclude_ymls=[new_yml_path, *exclude_ymls],
                    dataset_paths_key=dataset_paths_key,
                )

                if "classes" in spec:
                    if spec["classes"] != classes:
                        raise InvalidDatasetDefinitionFile(
                            f"classes mismatch in {spec['defn_path']}"
                        )

                DatasetDefinition._check_for_non_disjoint_sets(
                    literal_defns, child_specs
                )

                literal_defns.update(child_specs)

            elif "image_path" in spec and "label_path" in spec:
                # ez case
                literal_spec = LiteralSpecification.from_dict(spec)

                DatasetDefinition._check_for_non_disjoint_sets(
                    literal_defns, {literal_spec}
                )

                literal_defns.add(LiteralSpecification.from_dict(spec))

            else:
                # even easier case
                raise InvalidDatasetDefinitionFile(
                    f"Invalid spec in dataset_paths: {spec}"
                )

        # walrus operator :=
        if duplicates := literal_defns & exclude_specs:
            raise InvalidDatasetDefinitionFile(
                "duplicate literal definition found in exclude paths!\n"
                f"duplicates are: {duplicates}"
            )

        return literal_defns

    @staticmethod
    def _check_for_non_disjoint_sets(s1: Set, s2: Set) -> None:
        if intersection := s1 & s2:
            # duplicate literal definitions, or one of the literal definitions that we found
            # is in the exclude set. Report them!
            raise InvalidDatasetDefinitionFile(
                "duplicates found when trying to add s1 to s2\n"
                f"duplicates are: {intersection}"
            )

    @staticmethod
    def _load_thumbnails(
        classes: List[str], yaml_data: Dict[str, Any]
    ) -> Optional[Dict[str, Union[Path, List[Path]]]]:
        if "thumbnail_augmentation" in yaml_data:
            class_to_thumbnails = yaml_data["thumbnail_augmentation"]

            if not isinstance(class_to_thumbnails, dict):
                raise InvalidDatasetDefinitionFile(
                    "thumbnail_augmentation must map class names to paths to thumbnail "
                    "directories (e.g. `misc: /path/to/thumbnails/misc`)"
                )

            for k in class_to_thumbnails:
                if k not in classes:
                    raise InvalidDatasetDefinitionFile(
                        f"thumbnail_augmentation class {k} is not a valid class name"
                    )

            for k, v in class_to_thumbnails.items():
                if not isinstance(v, list):
                    class_to_thumbnails[k] = [Path(v)]

            return class_to_thumbnails

        return None

    @staticmethod
    def _check_dataset_paths(
        dataset_paths: Set[LiteralSpecification], prune: bool = False
    ) -> Set[LiteralSpecification]:
        to_prune: Set[LiteralSpecification] = set()
        for spec in dataset_paths:
            if not (
                spec.image_path.is_dir()
                and spec.label_path.is_dir()
                and len(list(spec.label_path.iterdir())) > 0
            ):
                if prune:
                    warnings.warn(
                        f"image_path or label_path do not lead to a directory, or there are no labels.\n"
                        f"image_path={spec.image_path}\n"
                        f"label_path={spec.label_path}\n"
                        f"will prune."
                    )
                    to_prune.add(spec)
                else:
                    raise FileNotFoundError(
                        f"image_path or label_path do not lead to a directory, or there are no labels.\n"
                        f"image_path={spec.image_path}\n"
                        f"label_path={spec.label_path}\n"
                    )
        return dataset_paths - to_prune

    @staticmethod
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
