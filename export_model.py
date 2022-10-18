#! /usr/bin/env python3

import sys
import argparse
import subprocess

import onnx
import onnxsim
import onnxruntime

import torch
import torchvision

import numpy as np

from model import YOGO
from pathlib import Path
from argparsers import export_parser
from dataloader import get_dataloader


"""
Learning from https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def do_vis(filename):
    raise NotImplementedError("This is currently broken, y'all :'( yell at axel!")

    import torchviz

    # FIXME: this is a hack, we should just create a fake label tensor)
    from yogo_loss import YOGOLoss

    dataloaders = get_dataloader("healthy_cell_dataset.yml", 1)
    DL = dataloaders["val"]

    Y = YOGO(0.1, 0.1)
    loss_fcn = YOGOLoss()

    img_batch, label_batch = next(iter(DL))

    out = Y(img_batch)
    g = torchviz.make_dot(
        loss_fcn(out, label_batch),
        params=dict(Y.named_parameters()),
        show_attrs=True,
        show_saved=True,
    )
    g.render(filename, format="pdf", view=False)


def do_export(args):
    pth_filename = args.input
    onnx_filename = (
        args.output_filename
        if args.output_filename is not None
        else pth_filename.replace("pth", "onnx")
    )

    if not onnx_filename.endswith(".onnx"):
        onnx_filename += ".onnx"

    model_save = torch.load(pth_filename, map_location=torch.device("cpu"))
    net = YOGO.from_pth(model_save, inference=True)
    net.eval()

    img_h, img_w = model_save["model_state_dict"]["img_size"]
    dummy_input = torch.randn(1, 1, img_h, img_w, requires_grad=False)
    torch_out = net(dummy_input)

    torch.onnx.export(net, dummy_input, onnx_filename, verbose=False, opset_version=14)

    # Load the ONNX model
    model_candidate = onnx.load(onnx_filename)

    if args.simplify:
        model_simplified_candidate, model_simplified_OK = onnxsim.simplify(
            model_candidate
        )
        model = model_simplified_candidate if model_simplified_OK else model_candidate
        simplify_msg = f" (simplified)"
    else:
        model = model_candidate
        simplify_msg = ""

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

    success_msg = f"exported to {onnx_filename}" + simplify_msg

    if args.IR:
        try:
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
            success_msg += f", {onnx_filename.replace('onnx', 'xml')}, {onnx_filename.replace('onnx', 'bin')}"
        except Exception as e:
            # if some error occurs, just quietly fail
            success_msg += f"; could not export to IR: {str(e)}"

    print("\n" * 3)
    print(success_msg)


if __name__ == "__main__":
    parser = export_parser()
    args = parser.parse_args()

    try:
        do_export(args)
    except AttributeError:
        parser.print_help()
