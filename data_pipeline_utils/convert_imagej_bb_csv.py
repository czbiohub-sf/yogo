#! /usr/bin/env python3

import sys
import csv

from pathlib import Path


from utils import convert_coords

""" This is a tool to convert imagej BBs to the YOGO format,
although this is most likely depricated in favour of cellpose.
"""


def process(fd, label_dir):
    reader = csv.DictReader(fd)
    for row in reader:
        img_name = Path(row["image_id"]).with_suffix(".csv")
        label_name = label_dir / img_name

        xcenter, ycenter, width, height = convert_coords(
            row["xmin"], row["xmax"], row["ymin"], row["ymax"]
        )

        label = row["label"]

        with open(label_name, "a") as current_file:
            current_file.write(f"{label},{xcenter},{ycenter},{width},{height}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <labels.csv> <label dir>")
        sys.exit(1)

    with open(sys.argv[1], "r") as fr:
        process(fr, sys.argv[2])
