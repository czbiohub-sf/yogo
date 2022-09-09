import argparse


def parse():
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
    parser.add_argument(
        "--sweep-id",
        type=str,
        nargs="?",
        help="set a sweep ID - if not doing a sweep, leave blank!",
    )
    return parser.parse_args()
