import onnx
import torch
import pytest
import subprocess
import onnxruntime

import numpy as np

from pathlib import Path

from yogo.model import YOGO
from yogo.utils.export_model import YOGOWrap, to_numpy


@pytest.fixture
def setup_yogo():
    y_raw = YOGO((772, 1032), 0.05, 0.05, 7, inference=True)
    y_raw.eval()

    y_wrap = YOGOWrap((772, 1032), 0.05, 0.05, 7, inference=True)
    y_wrap.load_state_dict(y_raw.state_dict())
    y_wrap.eval()

    img_h, img_w = y_wrap.img_size

    dummy_input = torch.randint(0, 256, (1, 1, int(img_h.item()), int(img_w.item())))

    onnx_filename = "onnx_out.onnx"

    yield y_raw, y_wrap, dummy_input, onnx_filename

    # Teardown (replaces tearDownClass)
    Path(onnx_filename).unlink(missing_ok=True)
    Path(onnx_filename).with_suffix(".bin").unlink(missing_ok=True)
    Path(onnx_filename).with_suffix(".xml").unlink(missing_ok=True)


@pytest.mark.parametrize(
    "name, normalize_images",
    [
        ("normalize_images", torch.tensor(True)),
        ("no_normalize_images", torch.tensor(False)),
    ],
)
@torch.no_grad()
def test_yogo_wrap(setup_yogo, name, normalize_images):
    y_raw, y_wrap, dummy_input, _ = setup_yogo
    y_wrap.normalize_images = normalize_images

    assert torch.allclose(
        y_raw(dummy_input.float() / 255.0 if normalize_images else dummy_input),
        y_wrap(dummy_input),
        rtol=1e-3,
        atol=1e-5,
    ), f"{name}={normalize_images} failed"


@pytest.mark.parametrize(
    "name, normalize_images",
    [
        ("normalize_images", torch.tensor(True)),
        ("no_normalize_images", torch.tensor(False)),
    ],
)
@torch.no_grad()
def test_torch_trace(setup_yogo, name, normalize_images):
    _, y_wrap, dummy_input, _ = setup_yogo
    y_wrap.normalize_images = normalize_images

    compiled = torch.jit.trace(y_wrap, dummy_input)
    compiled_out = compiled(dummy_input)
    torch_out = y_wrap(dummy_input)

    assert torch.allclose(
        torch_out,
        compiled_out,
        rtol=1e-3,
        atol=1e-5,
    ), f"{name} failed"


@pytest.mark.parametrize(
    "name, normalize_images",
    [
        ("normalize_images", torch.tensor(True)),
        ("no_normalize_images", torch.tensor(False)),
    ],
)
@torch.no_grad()
def test_onnx_export(setup_yogo, name, normalize_images):
    _, y_wrap, dummy_input, onnx_filename = setup_yogo
    y_wrap.normalize_images = normalize_images

    torch.onnx.export(
        y_wrap,
        dummy_input,
        onnx_filename,
        verbose=False,
        do_constant_folding=True,
        opset_version=17,
    ), f"{name} failed"

    model_candidate = onnx.load(onnx_filename)
    onnx.checker.check_model(model_candidate)

    # Compare model output from pure torch and onnx
    ort_session = onnxruntime.InferenceSession(str(onnx_filename))
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    torch_out = y_wrap(dummy_input)

    np.testing.assert_allclose(
        to_numpy(torch_out),
        ort_outs[0],
        rtol=1e-3,
        atol=1e-5,
        err_msg="pytorch and onnx outputs are far apart",
    )


@pytest.mark.parametrize(
    "name, normalize_images",
    [("normalize_images", True), ("no_normalize_images", False)],
)
@torch.no_grad()
def test_openvino_export(setup_yogo, name, normalize_images):
    _, _, _, onnx_filename = setup_yogo

    # export to IR
    subprocess.run(
        [
            "mo",
            "--input_model",
            str(onnx_filename),
            "--output_dir",
            Path(onnx_filename).resolve().parents[0],
            "--compress_to_fp16",
            "True",
        ],
        stdout=subprocess.DEVNULL,
    )
