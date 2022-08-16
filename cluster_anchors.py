#! /usr/bin/env python3

""" K-means clustering of anchors
"""

import glob
import numpy as np
import numpy.typing as npt

from collections import namedtuple


# [xmin, xmax, ymin, ymax]
Box = npt.NDArray[np.float64]


def xc_yc_w_h_to_corners(xc, yc, w, h):
    return (
        xc - w / 2,
        xc + w / 2,
        yc - h / 2,
        yc + h / 2,
    )


def corners_to_xc_yc_w_h(xmin, xmax, ymin, ymax):
    return (
        (xmax + xmin) / 2,
        (ymax + ymin) / 2,
        (xmax - xmin),
        (ymax - ymin),
    )


def area(b1: Box):
    return abs(b1[1] - b1[0]) * abs(b1[3] - b1[2])


def iou(b1: Box, b2: Box):
    inner_xmin = max(b1[0], b2[0])
    inner_xmax = min(b1[1], b2[1])
    inner_ymin = max(b1[2], b2[2])
    inner_ymax = min(b1[3], b2[3])

    if inner_xmax < inner_xmin or inner_ymax < inner_ymin:
        return 0

    intersection = area(np.array([inner_xmin, inner_xmax, inner_ymin, inner_ymax]))
    union = area(b1) + area(b2) - intersection
    return intersection / union


def get_all_bounding_boxes(bb_dir):
    bbs = []
    for fname in glob.glob(f"{bb_dir}/*.csv"):
        with open(fname, "r") as f:
            for line in f:
                vs = [float(v) for v in line.split(",")]
                bbs.append(xc_yc_w_h_to_corners(*vs[1:]))
    return np.array(bbs)



if __name__ == "__main__":
    import sys

    b1 = np.array([0, 1, 0, 1])
    print(area(b1))
    print(iou(b1, b1))
    print(get_all_bounding_boxes(sys.argv[1]).shape)
