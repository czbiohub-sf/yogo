#! /usr/bin/env python3

import sys
import zarr
import math

from PIL import Image
from tqdm import tqdm
from pathlib import Path


def convert(path_to_zarr_zip: str, out_dir: str):
    data = zarr.open(path_to_zarr_zip)
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    N = int(math.log(len(data), 10) + 1)

    for i in tqdm(range(len(data))):
        img = data[i][:]
        Image.fromarray(img).save(f"{out_dir}/img_{i:0{N}}.png")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <zarr zip file> <path to output dir>")
        sys.exit(1)

    convert(sys.argv[1], sys.argv[2])
