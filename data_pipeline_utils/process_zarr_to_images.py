#! /usr/bin/env python3

import sys
import zarr
import math

from PIL import Image
from pathlib import Path

from utils import multiprocess_directory_work


""" See README.md in this directory

This does part '1' of the "Operations on the data"
"""


def convert_zarr_to_image_folder(path_to_zarr_zip: Path):
    data = zarr.open(str(path_to_zarr_zip))

    image_dir = path_to_zarr_zip.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    N = int(math.log(len(data), 10) + 1)

    for i in range(len(data)):
        img = data[i][:]
        Image.fromarray(img).save(image_dir / f"img_{i:0{N}}.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path to run-sets directory>")
        sys.exit(1)

    run_set = Path(sys.argv[1])
    if not run_set.exists():
        raise FileNotFoundError(f"directory {sys.argv[1]} not found")

    files = list(run_set.glob("./**/*.zip"))

    if len(files) == 0:
        raise ValueError(f"no zarr files found in directory {sys.argv[1]}")

    multiprocess_directory_work(files, convert_zarr_to_image_folder)
