#! /usr/bin/env python3

import sys
import time
import glob

import zarr
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from functools import partial
from cellpose import utils, io

from utils import normalize, convert_coords, multiprocess_directory_work


def to_yogo_labels(label, output_dir_path, f):
    "depricated"
    outlines = load_cellpose_npy_file(f)
    file_name = Path(f).name.replace("_seg", "")
    new_csv = (output_dir_path / file_name).with_suffix(".csv")

    with open(new_csv, "w") as g:
        for outline in outlines:
            xmin, xmax, ymin, ymax = (
                outline[:, 0].min(),
                outline[:, 0].max(),
                outline[:, 1].min(),
                outline[:, 1].max(),
            )
            xcenter, ycenter, width, height = convert_coords(xmin, xmax, ymin, ymax)
            g.write(f"{label},{xcenter},{ycenter},{width},{height}\n")


def to_bb_labels(label, bb_csv_fd, f):
    outlines = load_cellpose_npy_file(f)
    file_path = Path(f)
    image_path_str = file_path.parent / file_path.name.replace("_seg", "")

    for outline in outlines:
        xmin, xmax, ymin, ymax = (
            outline[:, 0].min(),
            outline[:, 0].max(),
            outline[:, 1].min(),
            outline[:, 1].max(),
        )
        bb_csv_fd.write(f"{image_path_str},{xmin},{xmax},{ymin},{ymax}\n")


def load_cellpose_npy_file(f):
    data = np.load(f, allow_pickle=True).item()
    return utils.outlines_list(data["masks"])


def process_cellpose_results_to_yogo(files, output_dir, label=0):
    "depricated"
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    work_fcn = partial(to_yogo_labels, label, output_dir_path)

    multiprocess_directory_work(files, work_fcn)


def process_cellpose_results_to_bb_labels(files, bb_csv_path: Path, label=0):
    with open(str(bb_csv_path), "w") as bb_csv_fd:
        for f in files:
            to_bb_labels(label, bb_csv_fd, f)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            f"usage: {sys.argv[0]} <input dir> <label 0 (healthy, default), 1 (ring), 2 (troph), 3 (schizont)>"
        )
        sys.exit(1)

    try:
        label = sys.argv[3]
    except IndexError:
        label = "0"

    files = glob.glob(f"{sys.argv[1]}/*.npy")
    bb_csv = Path(sys.argv[1]).parent / "labels.csv"
    process_cellpose_results_to_bb_labels(files, bb_csv, label=label)
