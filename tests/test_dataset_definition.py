import pytest

from pathlib import Path

from yogo.data.dataset_definition_file import (
    DatasetDefinition,
    InvalidDatasetDefinitionFile,
)

"""
TODO test test_dataset_paths v. dataset_paths, make sure they play well together
"""


TEST_DIR = Path(__file__).parent
DEFNS_PATH = TEST_DIR / "fake-data" / "defns"


def test_basic_load() -> None:
    """should successfully load each and have one path per"""
    for basic_defn in ("literal_1.yml", "literal_2.yml", "literal_3.yml"):
        dataset_defn = DatasetDefinition.from_yaml(DEFNS_PATH / basic_defn)
        assert len(dataset_defn.dataset_paths) == 1
        assert len(dataset_defn.test_dataset_paths) == 0

    dataset_defn = DatasetDefinition.from_yaml(DEFNS_PATH / "literal_123.yml")
    assert len(dataset_defn.dataset_paths) == 3
    assert len(dataset_defn.test_dataset_paths) == 0


def test_basic_recursive_load_0() -> None:
    """
    loading a recursive defn that loads one literal defn
    should be equivalent to loading the literal defn
    """
    literal_defn = DatasetDefinition.from_yaml(DEFNS_PATH / "literal_1.yml")
    recursive_defn = DatasetDefinition.from_yaml(DEFNS_PATH / "recursive_1.yml")
    assert literal_defn == recursive_defn


def test_basic_recursive_load_1() -> None:
    """
    should be able to load both a recursive and a literal definition in one file
    """
    literal_defn_1 = DatasetDefinition.from_yaml(DEFNS_PATH / "literal_1.yml")
    literal_defn_2 = DatasetDefinition.from_yaml(DEFNS_PATH / "literal_2.yml")
    literal_concat = literal_defn_1 + literal_defn_2
    recursive_defn = DatasetDefinition.from_yaml(
        DEFNS_PATH / "recursive_1_literal_2.yml"
    )
    assert literal_concat == recursive_defn


def test_basic_recursive_load_2() -> None:
    """
    equality of dataset definitions is agnostic to order of the lists
    """
    recursive_12 = DatasetDefinition.from_yaml(DEFNS_PATH / "recursive_1_literal_2.yml")
    recursive_21 = DatasetDefinition.from_yaml(DEFNS_PATH / "recursive_2_literal_1.yml")
    assert recursive_12 == recursive_21


def test_basic_recursive_load_3() -> None:
    """
    should be able to recur more than once
    """
    recursive_123 = DatasetDefinition.from_yaml(DEFNS_PATH / "recursive_rec_123.yml")
    literal_123 = DatasetDefinition.from_yaml(DEFNS_PATH / "literal_123.yml")
    assert recursive_123 == literal_123


def test_recursive_cycle_0() -> None:
    """
    should detect a cycle
    """
    with pytest.raises(InvalidDatasetDefinitionFile):
        DatasetDefinition.from_yaml(DEFNS_PATH / "cycle_1.yml")

    # redundant, but oh well
    with pytest.raises(InvalidDatasetDefinitionFile):
        DatasetDefinition.from_yaml(DEFNS_PATH / "cycle_2.yml")

    # more redundant, but may as well!
    with pytest.raises(InvalidDatasetDefinitionFile):
        DatasetDefinition.from_yaml(DEFNS_PATH / "cycle_3.yml")


def test_recursive_cycle_1() -> None:
    """
    should detect a cycle
    """
    with pytest.raises(InvalidDatasetDefinitionFile):
        DatasetDefinition.from_yaml(DEFNS_PATH / "cycle_self.yml")


def test_unique_paths() -> None:
    """
    should detect duplicate paths
    """
    with pytest.raises(InvalidDatasetDefinitionFile):
        DatasetDefinition.from_yaml(DEFNS_PATH / "duplicate_paths.yml")


def test_path_check() -> None:
    """
    should detect missing paths
    """
    with pytest.raises(FileNotFoundError):
        DatasetDefinition.from_yaml(DEFNS_PATH / "literal-non-existant.yml")
