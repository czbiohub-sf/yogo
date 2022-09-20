import argparse


def train_parser():
    parser = argparse.ArgumentParser(description="commence a training run")
    parser.add_argument(
        "dataset_descriptor_file",
        type=str,
        help="path to yml dataset descriptor file",
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
    return parser.parse_args()


def export_parser():
    parser = argparse.ArgumentParser(description="convert a pth to onnx or Intel IR")

    subparsers = parser.add_subparsers()

    export_parser = subparsers.add_parser(
        "export", description="export PTH file to various formats"
    )
    export_parser.add_argument(
        "input",
        type=str,
        help="path to input pth file",
    )
    export_parser.add_argument(
        "--output-filename",
        type=str,
        help="output filename",
    )
    export_parser.add_argument(
        "--simplify",
        type=bool,
        help="attempt to simplify the onnx model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    export_parser.add_argument(
        "--IR",
        type=bool,
        help="export to IR (for NCS2)",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--visualize",
        type=bool,
        help="visualize PyTorch computational graph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser
