#! /usr/bin/env python3

import sys
import glob

import numpy as np

from tqdm import tqdm
from pathlib import Path

from _utils import multiprocess_directory_work


"""
for a 480 kB image input, cellpose saves a 16 MB .npy file!

It saves a copy of the original image (in RGB), two other full sized copies (for
masks and outlines), and a bunch of other metadata that we don't care about.

This should just overwrite those with the things we want - the segmentation!
"""


def work_fcn(f):
    data = np.load(f, allow_pickle=True).item()
    np.save(f, {"masks": data["masks"], "filename": data["filename"]})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path to dir of cellpose numpy files>")
        sys.exit(1)
    files = glob.glob(f"{sys.argv[1]}/*.npy")
    multiprocess_directory_work(files, work_fcn)
