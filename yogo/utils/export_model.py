#! /usr/bin/env python3

import subprocess

import onnx
import onnxsim
import onnxruntime

import torch

import numpy as np

from pathlib import Path

from yogo.model import YOGO
from yogo.utils.argparsers import export_parser


"""
Learning from https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def do_export(args):
    pth_filename = args.input
    onnx_filename = Path(
        args.output_filename
        if args.output_filename is not None
        else pth_filename.replace("pth", "onnx")
    ).with_suffix(".onnx")

    net, _ = YOGO.from_pth(pth_filename, inference=True)
    net.eval()

    img_h, img_w = net.img_size

    if args.crop_height is not None:
        img_h = (args.crop_height * img_h).round()

        crop_size = (img_h, img_w)
        Sx, Sy = net.get_grid_size(crop_size)
        _Cxs = torch.linspace(0, 1 - 1 / Sx, Sx).expand(Sy, -1)
        _Cys = (
            torch.linspace(0, 1 - 1 / Sy, Sy)
            .expand(1, -1)
            .transpose(0, 1)
            .expand(Sy, Sx)
        )

        net.register_buffer("_Cxs", _Cxs.clone())
        net.register_buffer("_Cys", _Cys.clone())

    dummy_input = torch.randn(
        1, 1, int(img_h.item()), int(img_w.item()), requires_grad=False
    )
    torch_out = net(dummy_input)

    torch.onnx.export(
        net,
        dummy_input,
        str(onnx_filename),
        verbose=False,
        do_constant_folding=True,
        opset_version=14,
    )

    # Load the ONNX model
    model_candidate = onnx.load(str(onnx_filename))

    if args.simplify:
        model_simplified_candidate, model_simplified_OK = onnxsim.simplify(
            model_candidate
        )
        model = model_simplified_candidate if model_simplified_OK else model_candidate
    else:
        model = model_candidate

    # Check that the model is well formed
    onnx.checker.check_model(model)

    # Compare model output from pure torch and onnx
    ort_session = onnxruntime.InferenceSession(str(onnx_filename))
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(
        to_numpy(torch_out),
        ort_outs[0],
        rtol=1e-3,
        atol=1e-5,
        err_msg="onnx and pytorch outputs are far apart",
    )

    success_msg = f"exported to {str(onnx_filename)}"

    # export to IR
    subprocess.run(
        [
            "mo",
            "--input_model",
            str(onnx_filename),
            "--output_dir",
            onnx_filename.resolve().parents[0],
            "--data_type",
            "FP16",
        ]
    )
    success_msg += f", {str(onnx_filename.with_suffix('.xml'))}, {str(onnx_filename.with_suffix('.bin'))}"

    print("\n")
    print(success_msg)


if __name__ == "__main__":
    parser = export_parser()
    args = parser.parse_args()

    try:
        do_export(args)
    except AttributeError:
        parser.print_help()
