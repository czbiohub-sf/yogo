import argparse


def parse():
    parser = argparse.ArgumentParser(description="commence a training run")
    parser.add_argument(
        "dataset_descriptor_file",
        type=str,
        nargs="?",
        help="path to yml dataset descriptor file",
        default="healthy_cell_dataset.yml",
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
