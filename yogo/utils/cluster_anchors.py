#! /usr/bin/env python3

""" K-means clustering of anchors
"""

from __future__ import annotations

import glob
import torch
import numpy as np
import numpy.typing as npt

from pathlib import Path
from typing import cast, Union, Tuple, Sequence, List


# [dims, xmin, xmax, ymin, ymax]
CornerBox = Union["npt.NDArray[np.float64]", torch.Tensor]
# [dims, xc, yc, w, h]
CenterBox = Union["npt.NDArray[np.float64]", torch.Tensor]
Box = Union[CornerBox, CenterBox]


def centers_to_corners(b: CenterBox) -> CornerBox:
    return np.array(
        (
            b[..., 0] - b[..., 2] / 2,
            b[..., 0] + b[..., 2] / 2,
            b[..., 1] - b[..., 3] / 2,
            b[..., 1] + b[..., 3] / 2,
        )
    ).T


def corners_to_centers(b: CornerBox) -> CenterBox:
    return np.array(
        (
            (b[..., 1] + b[..., 0]) / 2,
            (b[..., 3] + b[..., 2]) / 2,
            (b[..., 1] - b[..., 0]),
            (b[..., 3] - b[..., 2]),
        )
    ).T


def iou(b1: CornerBox, b2: CornerBox) -> "npt.NDArray[np.float64]":
    """b1, b2 of shape [1,d]"""

    def area(b: CornerBox) -> "npt.NDArray[np.float64]":
        return np.abs((b[..., 1] - b[..., 0]) * (b[..., 3] - b[..., 2]))

    intersection = np.maximum(
        np.minimum(b1[..., [1, 3]], b2[..., [1, 3]])
        - np.maximum(b1[..., [0, 2]], b2[..., [0, 2]]),
        0,
    ).prod(-1)
    return intersection / (area(b1) + area(b2) - intersection)


def gen_random_box(n=1, center_box=False) -> CornerBox:
    xmin = np.random.rand(n, 1) / 2
    xmax = np.random.rand(n, 1) / 2 + xmin
    ymin = np.random.rand(n, 1) / 2
    ymax = np.random.rand(n, 1) / 2 + ymin
    cb = np.hstack((xmin, xmax, ymin, ymax))
    if center_box:
        return corners_to_centers(cb)
    return cb


def get_dataset_bounding_boxes(
    bb_dirs: Sequence[Union[Path, str]], center_box=False
) -> Union[CenterBox, CornerBox]:
    return np.vstack(
        tuple(get_bounding_boxes(str(d), center_box=center_box) for d in bb_dirs)
    )


def get_bounding_boxes(bb_dir: str, center_box=False) -> Union[CenterBox, CornerBox]:
    def conv_func(x):
        return x if center_box else centers_to_corners

    bbs = []
    for fname in glob.glob(f"{bb_dir}/*.csv") + glob.glob(f"{bb_dir}/*.txt"):
        with open(fname, "r") as f:
            for line in f:
                if "," in line:
                    vs = np.array([float(v) for v in line.split(",")[1:]])
                else:
                    vs = np.array([float(v) for v in line.split(" ")[1:]])
                bbs.append(conv_func(vs))
    if len(bbs) == 0:
        print(bb_dir, "is empty!")
    return np.array(bbs)


def k_means(data: CornerBox, k=3) -> CornerBox:
    """
    https://blog.paperspace.com/speed-up-kmeans-numpy-vectorization-broadcasting-profiling/
    assumptions:
        - data is shape (num datapoints, 4)
        - data is normalized to [0,1]
    """

    def dist(b1: CornerBox, b2: CornerBox):
        return 1 - iou(b1[:, np.newaxis, :], b2[np.newaxis, :, :])

    def get_closest_mean(data, means):
        return np.argmin(dist(data, means), axis=1)

    means = np.concatenate([gen_random_box() for _ in range(k)], axis=0)

    boxes = []
    for _ in range(20):
        boxes.append(means.copy())
        mean_groups = get_closest_mean(data, means)

        for m in range(k):
            means[m] = data[mean_groups == m].mean(axis=0)  # type: ignore

        boxes.append(means.copy())

    return means


def _calculate_best_anchor(data: CenterBox) -> Tuple[float, float]:
    def centered_wh_iou(b1: CenterBox, b2: CenterBox):
        "get iou, assuming b1 and b2 are centerd on each other"
        intr = np.minimum(b1[..., 2], b2[..., 2]) * np.minimum(b1[..., 3], b2[..., 3])
        area1 = b1[..., 2] * b1[..., 3]
        area2 = b2[..., 2] * b2[..., 3]
        res = intr / (area1 + area2 - intr)
        return res

    def f(x: CenterBox):
        return (1 - centered_wh_iou(x, data)).sum()

    corners = k_means(centers_to_corners(data), k=1)[0]  # x y x y
    centers = corners_to_centers(corners)  # xc yc w h
    return cast(Tuple[float, float], (centers[2], centers[3]))


def best_anchor(label_paths: List[Union[Path, str]]) -> Tuple[float, float]:
    bbs = get_dataset_bounding_boxes(label_paths, center_box=True)
    anchor_w, anchor_h = _calculate_best_anchor(bbs)
    return anchor_w, anchor_h
