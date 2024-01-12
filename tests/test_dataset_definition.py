from pathlib import Path

from yogo.data.dataset_description_file import DatasetDefinition

# TODO need to make the data automatically generated? Or somehow
# otherwise deal with the absolute paths.
DEFNS_PATH = Path("/Users/axel.jacobsen/Desktop/fake/defns")


def test_basic_load():
    """should successfully load each and have one path per"""
    for basic_defn in ("literal_1.yml", "literal_2.yml", "literal_3.yml"):
        dataset_defn = DatasetDefinition.from_yaml(DEFNS_PATH / basic_defn)
        assert len(dataset_defn.dataset_paths) == 1
        assert len(dataset_defn.test_dataset_paths) == 0

    dataset_defn = DatasetDefinition.from_yaml(DEFNS_PATH / "literal_123.yml")
    assert len(dataset_defn.dataset_paths) == 3
    assert len(dataset_defn.test_dataset_paths) == 0


def test_basic_recursive_load_0():
    """
    loading a recursive defn that loads one literal defn
    should be equivalent to loading the literal defn
    """
    literal_defn = DatasetDefinition.from_yaml(DEFNS_PATH / "literal_1.yml")
    recursive_defn = DatasetDefinition.from_yaml(DEFNS_PATH / "recursive_1.yml")
    assert literal_defn == recursive_defn


def test_basic_recursive_load_1():
    """
    should be able to load both a recursive and a literal definition in one file
    """
    literal_defn_1 = DatasetDefinition.from_yaml(DEFNS_PATH / "literal_1.yml")
    literal_defn_2 = DatasetDefinition.from_yaml(DEFNS_PATH / "literal_2.yml")
    literal_concat = literal_defn_1 + literal_defn_2
    recursive_defn = DatasetDefinition.from_yaml(DEFNS_PATH / "recursive_1_literal_2.yml")
    assert literal_concat == recursive_defn

def test_basic_recursive_load_2():
    """
    equality of dataset definitions is agnostic to order of the lists
    """
    recursive_12 = DatasetDefinition.from_yaml(DEFNS_PATH / "recursive_1_literal_2.yml")
    recursive_21 = DatasetDefinition.from_yaml(DEFNS_PATH / "recursive_2_literal_1.yml")
    assert recursive_12 == recursive_21
