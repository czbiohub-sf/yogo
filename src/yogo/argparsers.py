import argparse

from pathlib import Path


try:
    boolean_action = argparse.BooleanOptionalAction  # type: ignore
except AttributeError:
    boolean_action = "store_true"  # type: ignore


def global_parser():
    parser = argparse.ArgumentParser(description="looking for a glance?")
    subparsers = parser.add_subparsers(help="here is what you can do", dest="task")
    train_parser(parser=subparsers.add_parser("train", help="train a model"))
    export_parser(parser=subparsers.add_parser("export", help="export a model"))
    infer_parser(
        parser=subparsers.add_parser("infer", help="infer images using a model")
    )
    return parser


def uint(val: int):
    try:
        v = int(val)
        if v >= 0:
            return v
    except ValueError:
        raise argparse.ArgumentTypeError(f"{val} is not a positive integer")


def train_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="commence a training run")

    parser.add_argument(
        "dataset_descriptor_file",
        type=str,
        help="path to yml dataset descriptor file",
    )
    parser.add_argument(
        "--from-pretrained",
        type=Path,
        help="start training from the provided pth file",
        default=None,
    )
    parser.add_argument(
        "--batch-size", type=uint, help="batch size for training", default=None
    )
    parser.add_argument(
        "--lr", type=float, help="learning rate for training", default=None
    )
    parser.add_argument(
        "--epochs", type=uint, help="number of epochs to train", default=None
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        const="adam",
        nargs="?",
        choices=["adam", "lion"],
        help="optimizer for training run",
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
    parser.add_argument(
        "--no-classify",
        default=False,
        action=boolean_action,
        help="turn off classification loss - good only for pretraining just a cell detector",
    )
    parser.add_argument(
        "--normalize-imgs",
        type=bool,
        default=False,
        help="normalize images into [0,1]"
    )

    image_resize_options = parser.add_mutually_exclusive_group(required=False)
    image_resize_options.add_argument(
        "--resize",
        type=int,
        nargs=2,
        help="resize image to these dimensions. e.g. '-r 300 400' to resize to width=300, height=400",
    )
    image_resize_options.add_argument(
        "--crop",
        type=float,
        help="crop image verically - i.e. '-c 0.25' will crop images to (round(0.25 * height), width)",
    )
    return parser


def export_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="convert a pth to onnx or Intel IR"
        )

    parser.add_argument(
        "input",
        type=str,
        help="path to input pth file",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        help="output filename",
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
    return parser
