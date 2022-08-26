#! /usr/bin/env python3

import sys
import subprocess

import onnx
import onnxruntime

import torch
import torchvision

import numpy as np

from pathlib import Path

from model import YOGO


"""
Learning from https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""

# TODO: onnxsim to simplify models?


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: ./to_onxx.py <your_file.pth> [<output_file_name.onnx>]")
        sys.exit(1)

    pth_filename = sys.argv[1]
    onnx_filename = (
        sys.argv[2] if len(sys.argv) == 3 else pth_filename.replace("pth", "onnx")
    )

    net = YOGO(17 / 300, 17 / 400)
    net.eval()

    # TODO CPU vs GPU vs whatever else?
    model_save = torch.load(pth_filename, map_location=torch.device("cpu"))
    net.load_state_dict(model_save["model_state_dict"])

    dummy_input = torch.randn(1, 1, 300, 400, requires_grad=False)
    torch_out = net(dummy_input)

    torch.onnx.export(net, dummy_input, onnx_filename, verbose=False, opset_version=14)

    # Load the ONNX model
    model = onnx.load(onnx_filename)

    # Check that the model is well formed
    onnx.checker.check_model(model)

    # Compare model output from pure torch and onnx
    ort_session = onnxruntime.InferenceSession(onnx_filename)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(
        to_numpy(torch_out),
        ort_outs[0],
        rtol=1e-3,
        atol=1e-5,
        err_msg="onnx and pytorch outputs are far apart",
    )

    # export to IR
    subprocess.run(
        [
            "mo",
            "--input_model",
            onnx_filename,
            "--output_dir",
            Path(onnx_filename).resolve().parents[0],
        ]
    )

    print(
        f"\nexported to {onnx_filename}, {onnx_filename.replace('onnx', 'xml')}, {onnx_filename.replace('onnx', 'bin')}"
    )
