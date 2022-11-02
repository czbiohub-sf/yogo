#! /usr/bin/env python3

import sys
import glob

import numpy as np

from pathlib import Path

from cellpose import utils, io

from _utils import normalize, convert_coords


def process_cellpose_results(input_dir, output_dir):
    for f in glob.glob("*.npy"):
        outlines = load_cellpose_npy_file(f)
        new_csv = (Path(output_dir) / f.replace("_seg", "")).with_suffix(".csv")
        with open(new_csv, "w") as g:
            for outline in outlines:
                xmin, xmax, ymin, ymax = (
                    outline[:, 0].min(),
                    outline[:, 0].max(),
                    outline[:, 1].min(),
                    outline[:, 1].max(),
                )
                xcenter, ycenter, width, height = convert_coords(xmin, xmax, ymin, ymax)
                g.write(f"0,{xcenter},{ycenter},{width},{height}\n")


def load_cellpose_npy_file(f):
    data = np.load(f, allow_pickle=True).item()
    return utils.outlines_list(data["masks"])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <input dir> <label dir>")

    process_cellpose_results(sys.argv[1], sys.argv[2])
