#! /usr/bin/env python3

""" K-means clustering of anchors
"""

from __future__ import annotations

import glob
import torch
import numpy as np
from pathlib import Path

try:
    import numpy.typing as npt
except ImportError:
    pass

from typing import cast, Union, Tuple, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# [dims, xmin, xmax, ymin, ymax]
CornerBox = Union["npt.NDArray[np.float64]", torch.Tensor]
# [dims, xc, yc, w, h]
CenterBox = Union["npt.NDArray[np.float64]", torch.Tensor]
Box = Union[CornerBox, CenterBox]


def centers_to_corners(b: CenterBox) -> CornerBox:
    if isinstance(b, np.ndarray):
        return np.array(
            (
                b[..., 0] - b[..., 2] / 2,
                b[..., 0] + b[..., 2] / 2,
                b[..., 1] - b[..., 3] / 2,
                b[..., 1] + b[..., 3] / 2,
            )
        ).T
    elif isinstance(b, torch.Tensor):
        return torch.vstack(
            (
                b[..., 0] - b[..., 2] / 2,
                b[..., 0] + b[..., 2] / 2,
                b[..., 1] - b[..., 3] / 2,
                b[..., 1] + b[..., 3] / 2,
            )
        ).T
    else:
        raise ValueError(
            f"b must be of type npt.NDArray or torch.Tensor: Got {type(b)}"
        )


def corners_to_centers(b: CornerBox) -> CenterBox:
    if isinstance(b, np.ndarray):
        return np.array(
            (
                (b[..., 1] + b[..., 0]) / 2,
                (b[..., 3] + b[..., 2]) / 2,
                (b[..., 1] - b[..., 0]),
                (b[..., 3] - b[..., 2]),
            )
        ).T
    elif isinstance(b, torch.Tensor):
        return torch.vstack(
            (
                (b[..., 1] + b[..., 0]) / 2,
                (b[..., 3] + b[..., 2]) / 2,
                (b[..., 1] - b[..., 0]),
                (b[..., 3] - b[..., 2]),
            ),
        ).T
    else:
        raise ValueError(
            f"b must be of type npt.NDArray or torch.Tensor: Got {type(b)}"
        )


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


def torch_iou(b1: CornerBox, b2: CornerBox) -> torch.Tensor:
    """
    b1, b2 of shape [1,d]
    """
    if not isinstance(b1, torch.Tensor) or not isinstance(b2, torch.Tensor):
        raise ValueError(
            f"b1 and b2 must be torch.Tensor, but are {type(b1)} {type(b2)}"
        )

    def area(b):
        return torch.abs((b[..., 1] - b[..., 0]) * (b[..., 3] - b[..., 2]))

    b1 = cast(torch.Tensor, b1)
    b2 = cast(torch.Tensor, b2)
    intersection = torch.clamp(
        torch.minimum(b1[..., [1, 3]], b2[..., [1, 3]])
        - torch.maximum(b1[..., [0, 2]], b2[..., [0, 2]]),
        min=0,
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


def plot_boxes(boxes, color_period=0) -> None:
    colors = ["r", "g", "b", "c", "m", "y", "k"]
    assert (
        0 <= color_period < len(colors)
    ), f"color_period must be in [0, {len(colors)})"

    _, ax = plt.subplots()
    current_axis = plt.gca()
    for i, box in enumerate(boxes):
        color = colors[i % color_period if color_period > 0 else 0]
        _, _, w, h = corners_to_centers(box)
        current_axis.add_patch(
            Rectangle(
                (box[0], box[2]),
                w,
                h,
                facecolor=color if i > len(boxes) - 1 - color_period else "none",
                edgecolor=color,
            )
        )
    plt.show()


def get_dataset_bounding_boxes(
    bb_dirs: Sequence[Union[Path, str]], center_box=False
) -> Union[CenterBox, CornerBox]:
    return np.vstack(
        tuple(get_bounding_boxes(str(d), center_box=center_box) for d in bb_dirs)
    )


def get_bounding_boxes(bb_dir: str, center_box=False) -> Union[CenterBox, CornerBox]:
    conv_func = lambda x: x if center_box else centers_to_corners
    bbs = []
    for fname in glob.glob(f"{bb_dir}/*.csv"):
        with open(fname, "r") as f:
            for line in f:
                vs = np.array([float(v) for v in line.split(",")[1:]])
                bbs.append(conv_func(vs))
    return np.array(bbs)


def k_means(data: CornerBox, k=3, plot=False) -> CornerBox:
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
    for _ in range(100):
        boxes.append(means.copy())
        mean_groups = get_closest_mean(data, means)

        for m in range(k):
            means[m] = data[mean_groups == m].mean(axis=0)

        boxes.append(means.copy())

    if plot:
        plot_boxes(np.array(boxes).reshape(-1, 4), color_period=k)

    return means


def best_anchor(data: CenterBox, kmeans=False) -> Tuple[float, float]:
    """
    Optimization for k_means(data, k=1)

    FIXME: doesn't seem to work for a dataset with larger RBCs
    """
    import logging

    def centered_wh_iou(b1: CenterBox, b2: CenterBox):
        "get iou, assuming b1 and b2 are centerd on eachother"
        intr = np.minimum(b1[..., 2], b2[..., 2]) * np.minimum(b1[..., 3], b2[..., 3])
        area1 = b1[..., 2] * b1[..., 3]
        area2 = b2[..., 2] * b2[..., 3]
        res = intr / (area1 + area2 - intr)
        return res

    def f(x: CenterBox):
        return (1 - centered_wh_iou(x, data)).sum()

    if not kmeans:
        from scipy import optimize

        random_center_box = gen_random_box(center_box=True)[0]
        res = optimize.minimize(f, method="Nelder-Mead", x0=random_center_box)
        if res.success:
            return res.x[2], res.x[3]
        else:
            logging.warning(
                f"scipy could not optimize to ideal solution: '{res.message}'\n"
                f"defaulting to k_mean(data, k=1)"
            )
    corners = k_means(centers_to_corners(data), k=1)[0]
    centers = corners_to_centers(corners)
    return cast(Tuple[float, float], (centers[2], centers[3]))
