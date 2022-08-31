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
        help="note for the run (e.g. run on a TI-82)",
        default="",
    )
    return parser.parse_args()
