import argparse

from pathlib import Path

from yogo import DefaultHyperparams as df


try:
    boolean_action = argparse.BooleanOptionalAction  # type: ignore
except AttributeError:
    boolean_action = "store_true"  # type: ignore


def uint(val: int):
    try:
        v = int(val)
        if v >= 0:
            return v
    except ValueError:
        raise argparse.ArgumentTypeError(f"{val} is not a positive integer")


def global_parser():
    parser = argparse.ArgumentParser(description="looking for a glance?")
    subparsers = parser.add_subparsers(help="here is what you can do", dest="task")
    train_parser(parser=subparsers.add_parser("train", help="train a model"))
    export_parser(parser=subparsers.add_parser("export", help="export a model"))
    infer_parser(
        parser=subparsers.add_parser("infer", help="infer images using a model")
    )
    return parser


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
        "--batch-size",
        type=uint,
        help=f"batch size for training (default {df.BATCH_SIZE})",
        default=None,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help=f"learning rate for training (default {df.LEARNING_RATE})",
        default=None,
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        help=f"factor by which to decay lr - e.g. '2' will give a final learning rate of `lr` / 2 (default {df.DECAY_FACTOR})",
        default=None,
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        help=f"label smoothing - default 0.01 (default {df.LABEL_SMOOTHING})",
        default=0.01,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        help=f"weight decay for training (default {df.WEIGHT_DECAY})",
        default=None,
    )
    parser.add_argument(
        "--epochs",
        type=uint,
        help=f"number of epochs to train (default {df.EPOCHS})",
        default=None,
    )
    parser.add_argument(
        "--model",
        default=None,
        const=None,
        nargs="?",
        choices=[
            "base_model",
            "model_no_dropout",
            "model_smaller_SxSy",
            "model_big_simple",
            "model_big_normalized",
            "model_big_heavy_normalized",
        ],
        help="model version to use - do not use with --from-pretrained, as we use the pretrained model",
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        const="adam",
        nargs="?",
        choices=["adam", "lion"],
        help=f"optimizer for training run (default {df.OPTIMIZER_TYPE})",
    )
    parser.add_argument(
        "--note",
        type=str,
        help="note for the run (e.g. 'run on a TI-82')",
        default=None,
    )
    parser.add_argument(
        "--name",
        type=str,
        help="name for the run (e.g. 'ti-82_run')",
        default=None,
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
        help="turn off classification loss - good only for pretraining just a cell detector (default False)",
    )
    parser.add_argument(
        "--normalize-images",
        default=False,
        action=boolean_action,
        help="normalize images into [0,1] (default False)",
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
        parser = argparse.ArgumentParser(description="infer on image data")

    parser.add_argument(
        "pth_path", type=Path, help="path to .pth file defining the model"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="path to directory for results - ignore to not save results",
    )
    parser.add_argument(
        "--draw-boxes",
        help="plot and display each image",
        action=boolean_action,
        default=False,
    )
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument(
        "--path-to-images", type=Path, default=None, help="path to image or images"
    )
    data_source.add_argument(
        "--path-to-zarr", type=Path, default=None, help="path to zarr file"
    )
    return parser
