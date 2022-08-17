#! /usr/bin/env python3

""" K-means clustering of anchors
"""

import glob
import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
    return np.abs((b1[..., 1] - b1[..., 0]) * (b1[..., 3] - b1[..., 2]))


def iou(b1: Box, b2: Box):
    """b1, b2 of shape [1,d]"""
    intersection = np.maximum(
        np.minimum(b1[..., [1, 3]], b2[..., [1, 3]])
        - np.maximum(b1[..., [0, 2]], b2[..., [0, 2]]),
        0,
    ).prod(-1)
    return intersection / (area(b1) + area(b2) - intersection)


def get_all_bounding_boxes(bb_dir):
    bbs = []
    for fname in glob.glob(f"{bb_dir}/*.csv"):
        with open(fname, "r") as f:
            for line in f:
                vs = [float(v) for v in line.split(",")]
                bbs.append(xc_yc_w_h_to_corners(*vs[1:]))
    return np.array(bbs)


def gen_random_box():
    xmin = np.random.rand() / 2
    xmax = np.random.rand() / 2 + xmin
    ymin = np.random.rand() / 2
    ymax = np.random.rand() / 2 + ymin
    return np.array((xmin, xmax, ymin, ymax)).reshape(1, -1)


def plot_boxes(boxes, color_period=0):
    colors = ["r", "g", "b", "c", "m", "y", "k"]
    assert (
        0 <= color_period < len(colors)
    ), f"color_period must be in [0, {len(colors)})"

    _, ax = plt.subplots()
    current_axis = plt.gca()
    for i, box in enumerate(boxes):
        color = colors[i % color_period if color_period > 0 else 0]
        _, _, w, h = corners_to_xc_yc_w_h(*box)
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


def k_means(data, k=3, plot=False):
    """
    https://blog.paperspace.com/speed-up-kmeans-numpy-vectorization-broadcasting-profiling/
    assumptions:
        - data is shape (num datapoints, 4)
        - data is normalized to [0,1]
    """

    def dist(b1: Box, b2: Box):
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

    return corners_to_xc_yc_w_h(means)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path to labels>")
        sys.exit(1)

    data = get_all_bounding_boxes(sys.argv[1])
    # sanity checks for our data
    assert np.all(data[:, 0] < data[:, 1])
    assert np.all(data[:, 2] < data[:, 3])
    k_means(data, k=6, plot=True)
