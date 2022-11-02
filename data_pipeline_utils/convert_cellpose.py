#! /usr/bin/env python3

import sys
import time
import glob

import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from functools import partial
from cellpose import utils, io

from _utils import normalize, convert_coords


def process_cellpose_results(input_dir, output_dir, label=0):
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    work_fnc = partial(work, label, output_dir_path)

    files = glob.glob(f"{input_dir}/*.npy")
    cpu_count = mp.cpu_count()

    print(f"processing {len(files)} files")
    print(f"num cpus: {cpu_count}")

    with mp.Pool(cpu_count) as P:
        # list so we get tqdm output, thats it!
        for _ in tqdm(
            P.imap_unordered(work_fnc, files, chunksize=64), total=len(files)
        ):
            pass
        P.close()
        P.join()


def work(label, output_dir_path, f):
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


def load_cellpose_npy_file(f):
    data = np.load(f, allow_pickle=True).item()
    return utils.outlines_list(data["masks"])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <input dir> <label dir>")
        sys.exit(1)

    try:
        label = sys.argv[3]
    except IndexError:
        label = 0

    process_cellpose_results(sys.argv[1], sys.argv[2], label=label)
