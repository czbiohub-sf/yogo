#! /usr/bin/env python3

""" K-means clustering of anchors
"""

import glob
import torch
import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from typing import cast, Union, Tuple

# [dims, xmin, xmax, ymin, ymax]
CornerBox = Union[npt.NDArray[np.float64], torch.Tensor]
# [dims, xc, yc, w, h]
CenterBox = Union[npt.NDArray[np.float64], torch.Tensor]
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


def iou(b1: CornerBox, b2: CornerBox) -> npt.NDArray[np.float64]:
    """b1, b2 of shape [1,d]"""

    def area(b: CornerBox) -> npt.NDArray[np.float64]:
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

    def area(b):
        return torch.abs((b[..., 1] - b[..., 0]) * (b[..., 3] - b[..., 2]))

    intersection = torch.clamp(
        torch.minimum(b1[..., [1, 3]], b2[..., [1, 3]])
        - torch.maximum(b1[..., [0, 2]], b2[..., [0, 2]]),
        min=0,
    ).prod(-1)
    return intersection / (area(b1) + area(b2) - intersection)


def gen_random_box(n = 1, center_box=False) -> CornerBox:
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


def get_all_bounding_boxes(bb_dir, conv_to_corners=True) -> npt.NDArray[np.float64]:
    conv_func = centers_to_corners if conv_to_corners else lambda x: x
    bbs = []
    for fname in glob.glob(f"{bb_dir}/*.csv"):
        with open(fname, "r") as f:
            for line in f:
                vs = np.array([float(v) for v in line.split(",")])
                bbs.append(centers_to_corners(vs[1:]))
    return np.array(bbs)


def k_means(data, k=3, plot=False) -> CornerBox:
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
    for _ in range(50):
        boxes.append(means.copy())
        mean_groups = get_closest_mean(data, means)

        for m in range(k):
            means[m] = data[mean_groups == m].mean(axis=0)

        boxes.append(means.copy())

    if plot:
        plot_boxes(np.array(boxes).reshape(-1, 4), color_period=k)

    return cast(CornerBox, corners_to_centers(means))


def best_anchor(data: CenterBox) -> Tuple[float,float]:
    """Optimization for k_means(data, k=1)"""
    from scipy import optimize

    def centered_wh_iou(b1: CenterBox, b2: CenterBox):
        "get iou, assuming b1 and b2 are centerd on eachother"
        intr = np.minimum(b1[..., 2], b2[..., 2]) * np.minimum(b1[..., 3], b2[..., 3])
        area1 = b1[..., 2] * b1[..., 3]
        area2 = b2[..., 2] * b2[..., 3]
        res = intr / (area1 + area2 - intr)
        return res

    def f(x: CenterBox):
        return (1 - centered_wh_iou(x, data)).sum()

    res = optimize.minimize(f, method="Nelder-Mead", x0=gen_random_box(center_box=True))
    if res.success:
        return res.x[2], res.x[2]
    else:
        # FIXME: Logging?
        print(
            f"scipy could not optimize to ideal solution: '{res.message}'\n"
            f"defaulting to k_mean(data, k=1)"
        )
        corners = k_means(centers_to_corners(data), k=1)[0]
        centers = centers_to_corners(corners)
        return cast(Tuple[float,float], (centers[2], centers[3]))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path to labels>")
        sys.exit(1)

    data = get_all_bounding_boxes(sys.argv[1])
    # sanity checks for our data
    assert np.all(data[:, 0] < data[:, 1])
    assert np.all(data[:, 2] < data[:, 3])
    print(k_means(data, k=1, plot=True))
