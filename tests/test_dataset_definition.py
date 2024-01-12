from pathlib import Path

from yogo.data.dataset_description_file import DatasetDefinition


# @pytest.fixture
# def setup_yogo():
#     y_raw = YOGO((772, 1032), 0.05, 0.05, 7, inference=True)
#     y_raw.eval()

#     y_wrap = YOGOWrap((772, 1032), 0.05, 0.05, 7, inference=True)
#     y_wrap.load_state_dict(y_raw.state_dict())
#     y_wrap.eval()

#     img_h, img_w = y_wrap.img_size

#     dummy_input = torch.randint(0, 256, (1, 1, int(img_h.item()), int(img_w.item())))

#     onnx_filename = "onnx_out.onnx"

#     yield y_raw, y_wrap, dummy_input, onnx_filename

#     # Teardown (replaces tearDownClass)
#     Path(onnx_filename).unlink(missing_ok=True)
#     Path(onnx_filename).with_suffix(".bin").unlink(missing_ok=True)
#     Path(onnx_filename).with_suffix(".xml").unlink(missing_ok=True)


# @pytest.mark.parametrize(
#     "name, normalize_images",
#     [("normalize_images", True), ("no_normalize_images", False)],
# )


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


def test_basic_recursive_load():
    """
    loading a recursive defn that loads one literal defn
    should be equivalent to loading the literal defn
    """
    literal_defn = DatasetDefinition.from_yaml(DEFNS_PATH / "literal_1.yml")
    recursive_defn = DatasetDefinition.from_yaml(DEFNS_PATH / "recursive_1.yml")
    assert literal_defn == recursive_defn
