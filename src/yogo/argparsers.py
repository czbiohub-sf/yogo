import argparse

from typing import Union


try:
    boolean_action = argparse.BooleanOptionalAction  # type: ignore
except AttributeError:
    boolean_action = "store_true"  # type: ignore


def global_parser():
    parser = argparse.ArgumentParser(description="looking for a glance?")
    subparsers = parser.add_subparsers(help="here is what you can do", dest="task")
    train_subparser = train_parser(
        parser=subparsers.add_parser("train", help="train a model")
    )
    export_subparser = export_parser(
        parser=subparsers.add_parser("export", help="export a model")
    )
    infer_subparser = infer_parser(
        parser=subparsers.add_parser("infer", help="infer images using a model")
    )
    return parser


def train_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="commence a training run")

    parser.add_argument(
        "dataset_descriptor_file", type=str, help="path to yml dataset descriptor file",
    )
    parser.add_argument(
        "--note",
        type=str,
        nargs="?",
        help="note for the run (e.g. 'run on a TI-82')",
        default="",
    )
    parser.add_argument(
        "--group",
        type=str,
        nargs="?",
        help="group that the run belongs to (e.g. 'mAP test')",
    )
    parser.add_argument(
        "--device",
        type=str,
        nargs="?",
        help="set a device for the run - if not specified, we will try to use 'cuda', and fallback on 'cpu'",
    )
    return parser


def export_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="convert a pth to onnx or Intel IR"
        )

    parser.add_argument(
        "input", type=str, help="path to input pth file",
    )
    parser.add_argument(
        "--output-filename", type=str, help="output filename",
    )
    parser.add_argument(
        "--simplify",
        help="attempt to simplify the onnx model",
        action=boolean_action,
        default=True,
    )
    return parser


def infer_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="infer results over some dataset")

    parser.add_argument(
        "pth_path", type=str, help="path to .pth file defining the model"
    )
    parser.add_argument("images", type=str, help="path to image or images")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="path to directory for results - ignore to not save results",
        default=None,
    )
    parser.add_argument(
        "--visualize",
        help="plot and display each image",
        action=boolean_action,
        default=False,
    )
