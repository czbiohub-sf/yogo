import argparse

from pathlib import Path

from yogo.model_defns import MODELS
from yogo.utils.default_hyperparams import DefaultHyperparams as df


try:
    boolean_action = argparse.BooleanOptionalAction  # type: ignore
except AttributeError:
    boolean_action = "store_true"  # type: ignore


def uint(val: int):
    try:
        v = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{val} is not a positive integer")

    if v < 0:
        raise argparse.ArgumentTypeError(f"{val} is not a positive integer")

    return v


def super_unitary_float(val: float):
    "cheeky name for a number greater than or equal to 1"
    try:
        v = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{v} is not a float value")

    if not 1 <= v:
        raise argparse.ArgumentTypeError(f"{v} must be greater than or equal to 1")

    return v


def unsigned_float(val: float):
    try:
        v = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{v} is not a float value")

    if not (0 <= v):
        raise argparse.ArgumentTypeError(f"{v} must be greater than 0")

    return v


def unitary_float(val: float):
    try:
        v = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{v} is not a float value")

    if not (0 <= v <= 1):
        raise argparse.ArgumentTypeError(f"{v} must be in [0,1]")

    return v


def global_parser():
    parser = argparse.ArgumentParser(
        description="what can yogo do for you today?", allow_abbrev=False
    )
    subparsers = parser.add_subparsers(help="here is what you can do", dest="task")
    train_parser(
        parser=subparsers.add_parser("train", help="train a model", allow_abbrev=False)
    )
    export_parser(
        parser=subparsers.add_parser(
            "export", help="export a model", allow_abbrev=False
        )
    )
    infer_parser(
        parser=subparsers.add_parser(
            "infer", help="infer images using a model", allow_abbrev=False
        )
    )
    return parser


def train_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="commence a training run", allow_abbrev=False
        )

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
        "-bs",
        "--batch-size",
        type=uint,
        help=f"batch size for training (default: {df.BATCH_SIZE})",
        default=df.BATCH_SIZE,
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        "--lr",
        type=unitary_float,
        help=f"learning rate for training (default: {df.LEARNING_RATE})",
        default=df.LEARNING_RATE,
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=super_unitary_float,
        help=f"factor by which to decay lr - e.g. '2' will give a final learning rate of `lr` / 2 (default: {df.DECAY_FACTOR})",
        default=df.DECAY_FACTOR,
    )
    parser.add_argument(
        "--label-smoothing",
        type=unitary_float,
        help=f"label smoothing (default: {df.LABEL_SMOOTHING})",
        default=df.LABEL_SMOOTHING,
    )
    parser.add_argument(
        "-wd",
        "--weight-decay",
        type=unitary_float,
        help=f"weight decay for training (default: {df.WEIGHT_DECAY})",
        default=df.WEIGHT_DECAY,
    )
    parser.add_argument(
        "--epochs",
        type=uint,
        help=f"number of epochs to train (default: {df.EPOCHS})",
        default=df.EPOCHS,
    )
    parser.add_argument(
        "--no-obj-weight",
        type=float,
        help=f"weight for the objectness loss when there isn't an object (default: {df.NO_OBJ_WEIGHT})",
        default=df.NO_OBJ_WEIGHT,
    )
    parser.add_argument(
        "--iou-weight",
        type=float,
        help=f"weight for the iou loss (default: {df.IOU_WEIGHT})",
        default=df.IOU_WEIGHT,
    )
    parser.add_argument(
        "--classify-weight",
        type=float,
        help=f"weight for the classification loss (default: {df.CLASSIFY_WEIGHT})",
        default=df.CLASSIFY_WEIGHT,
    )
    parser.add_argument(
        "--healthy-weight",
        type=unitary_float,
        help=f"weight for healthy class, between 0 and 1 (default: {df.HEALTHY_WEIGHT})",
        default=df.HEALTHY_WEIGHT,
    )
    parser.add_argument(
        "--no-classify",
        default=False,
        action=boolean_action,
        help="turn off classification loss - good only for pretraining just a cell detector",
    )
    parser.add_argument(
        "--normalize-images",
        default=False,
        action=boolean_action,
        help="normalize images into [0,1]",
    )
    parser.add_argument(
        "--image-shape",
        default=(772, 1032),
        nargs=2,
        type=int,
        help="size of images for training (e.g. --image-shape 772 1032) (default: 772 1032)",
    )
    parser.add_argument(
        "--model",
        default=None,
        const=None,
        nargs="?",
        choices=list(MODELS.keys()),
        help="model version to use - do not use with --from-pretrained, as we use the pretrained model",
    )
    parser.add_argument(
        "--device",
        type=str,
        nargs="?",
        help="set a device for the run - if not specified, we will try to use 'cuda', and fallback on 'cpu'",
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
        "--tag",
        type=str,
        help="tag for the run (e.g. 'test')",
        default=None,
    )
    return parser


def export_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="convert a pth to onnx or Intel IR", allow_abbrev=False
        )

    parser.add_argument(
        "input",
        type=str,
        help="path to input pth file",
    )
    parser.add_argument(
        "--crop-height",
        type=unitary_float,
        help="crop image verically - '-c 0.25' will crop images to (round(0.25 * height), width)",
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
        parser = argparse.ArgumentParser(
            description="infer on image data", allow_abbrev=False
        )

    parser.add_argument(
        "pth_path", type=Path, help="path to .pth file defining the model"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="path to directory for results, either --draw-boxes or --save-preds",
    )
    parser.add_argument(
        "--draw-boxes",
        help="plot and either save (if --output-dir is set) or show each image",
        action=boolean_action,
        default=False,
    )
    parser.add_argument(
        "--save-preds",
        help=(
            "save predictions in YOGO label format - requires `--output-dir` "
            " to be set"
        ),
        action=boolean_action,
        default=False,
    )
    parser.add_argument(
        "--count",
        action=boolean_action,
        default=False,
        help="display the final predicted counts per-class",
    )
    parser.add_argument(
        "--batch-size",
        type=uint,
        help="batch size for inference (default: 64)",
        default=64,
    )
    parser.add_argument(
        "--device",
        type=str,
        nargs="?",
        help="set a device for the run - if not specified, we will try to use 'cuda', and fallback on 'cpu'",
    )
    parser.add_argument(
        "--crop-height",
        type=unitary_float,
        help="crop image verically - '-c 0.25' will crop images to (round(0.25 * height), width)",
    )
    parser.add_argument(
        "--output-img-filetype",
        type=str,
        choices=[".png", ".tif", ".tiff"],
        default=".png",
        help="filetype for output images (default: .png)",
    )
    parser.add_argument(
        "--obj-thresh",
        type=unsigned_float,
        default=0.5,
        help="objectness threshold for predictions (default: 0.5)",
    )
    parser.add_argument(
        "--iou-thresh",
        type=unsigned_float,
        default=0.5,
        help="intersection over union threshold for predictions (default: 0.5)",
    )
    parser.add_argument(
        "--min-class-confidence-threshold",
        type=unitary_float,
        default=0.0,
        help=(
            "minimum confidence for a class to be considered - i.e. the "
            "max confidence must be greater than this value (default: 0.0)"
        ),
    )
    parser.add_argument(
        "--heatmap-mask-path",
        type=Path,
        default=None,
        help="path to heatmap mask for the run (default: None)",
    )
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument(
        "--path-to-images", type=Path, default=None, help="path to image or images"
    )
    data_source.add_argument(
        "--path-to-zarr", type=Path, default=None, help="path to zarr file"
    )
    return parser
