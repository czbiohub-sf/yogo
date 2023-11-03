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


class YOGOWrap(YOGO):
    """
    two reasons to make this wrap:
        - we can normalize images within YOGO here, so we don't have to do it in ulc-malaria-scope
        - for some reason onnx likes `torch.split` but doesn't like `torch.chunk`, and torch.jit
          likes `torch.chunk` but doesn't like `torch.split`.
    So we wrap YOGO and use the version of forward that onnx likes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_images = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # made this more terse than YOGO's forward
        x = x.float()
        if self.normalize_images:
            x /= 255.0
        x = self.model(x)

        _, _, Sy, Sx = x.shape

        classification = torch.softmax(x[:, 5:, :, :], dim=1)

        clamped_whs = torch.clamp(x[:, 2:4, :, :], max=80)

        return torch.cat(
            (
                (1 / Sx) * torch.sigmoid(x[:, 0:1, :, :]) + self._Cxs,
                (1 / Sy) * torch.sigmoid(x[:, 1:2, :, :]) + self._Cys,
                self.anchor_w * torch.exp(clamped_whs[:, 0:1, :, :]),
                (
                    self.anchor_h
                    * torch.exp(clamped_whs[:, 1:2, :, :])
                    * self.height_multiplier
                ),
                torch.sigmoid(x[:, 4:5, :, :]),
                *torch.split(classification, 1, dim=1),
            ),
            dim=1,
        )


def do_export(args):
    pth_filename = args.input
    onnx_filename = Path(
        args.output_filename
        if args.output_filename is not None
        else pth_filename.replace("pth", "onnx")
    ).with_suffix(".onnx")

    net, cfg = YOGOWrap.from_pth(pth_filename, inference=True)
    net.normalize_images = cfg["normalize_images"]
    net.eval()

    img_h, img_w = net.img_size

    if args.crop_height is not None:
        img_h = (args.crop_height * img_h).round()
        net.resize_model(img_h.item())

    dummy_input = torch.randint(0, 256, (1, 1, int(img_h.item()), int(img_w.item())))

    torch.onnx.export(
        net,
        dummy_input,
        str(onnx_filename),
        verbose=False,
        do_constant_folding=True,
        opset_version=17,
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

    torch_out = net(dummy_input)
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
            "--compress_to_fp16",
            "True",
        ]
    )
    success_msg += f", {str(onnx_filename.with_suffix('.xml'))}, {str(onnx_filename.with_suffix('.bin'))}"

    print(success_msg)


if __name__ == "__main__":
    parser = export_parser()
    args = parser.parse_args()

    try:
        do_export(args)
    except AttributeError:
        parser.print_help()
