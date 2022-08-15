#! /usr/bin/env python3

""" K-means clustering of anchors
"""

from collections import namedtuple


Box = namedtuple("Box", ["xmin", "ymin", "xmax", "ymax"])


def area(b1: Box):
    return abs(b1.xmax - b1.xmin) * abs(b1.ymax - b1.ymin)


def iou(b1: Box, b2: Box):
    inner_xmin = max(b1.xmin, b2.xmin)
    inner_xmax = min(b1.xmax, b2.xmax)
    inner_ymin = max(b1.ymin, b2.ymin)
    inner_ymax = min(b1.ymax, b2.ymax)

    if inner_xmax < inner_xmin or inner_ymax < inner_ymin:
        return 0

    intersection = area(Box(inner_xmin, inner_ymin, inner_xmax, inner_ymax))
    union = area(b1) + area(b2) - intersection
    return intersection / union





if __name__ == "__main__":
    b1 = Box(xmin=0, ymin=0, xmax=1, ymax=1)
    print(area(b1))
    print(iou(b1, b1))
