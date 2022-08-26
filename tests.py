#! /usr/bin/env python3

import unittest
import numpy as np

from cluster_anchors import *
from yogo_loss import split_labels_into_bins


class TestClustering(unittest.TestCase):
    """
    it is best to draw out the bounding boxes for these tests
    """

    def test_IOU_sanity_checks(self) -> None:
        b1 = np.array([0, 1, 0, 1])
        b2 = np.array([1, 2, 1, 2])
        b3 = np.array([10, 11, 10, 11])
        self.assertEqual(iou(b1, b1), 1.0)
        self.assertEqual(iou(b2, b2), 1.0)
        self.assertEqual(iou(b1, b2), 0)
        self.assertEqual(iou(b1, b3), 0)

    def test_IOU_basic(self) -> None:
        b1 = np.array([0, 4, 2, 6])
        b2 = np.array([2, 6, 0, 4])
        self.assertEqual(iou(b1, b2), 4 / (4 * 4 + 4 * 4 - 4))
        self.assertEqual(iou(b2, b1), 4 / (4 * 4 + 4 * 4 - 4))

    def test_box_definition_conversions(self) -> None:
        for i in range(100):
            corners = np.random.rand(6, 4)
            self.assertTrue(
                np.allclose(
                    corners, xc_yc_w_h_to_corners(corners_to_xc_yc_w_h(corners))
                )
            )


class TestLossUtilities(unittest.TestCase):
    def test_labels_into_bins_1(self):
        labels = [
            [0.0, 0.1, 0.1, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0, 0.0],
        ]
        Sx, Sy = 2, 1
        d = split_labels_into_bins(labels, Sx, Sy)
        self.assertEqual(d[0, 0], [[0.0, 0.1, 0.1, 0.0, 0.0]])
        self.assertEqual(d[1, 0], [[0.0, 0.9, 0.1, 0.0, 0.0]])

    def test_labels_into_bins_2(self):
        labels = [
            [0.0, 0.1, 0.1, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0, 0.0],
            [0.0, 0.1, 0.9, 0.0, 0.0],
            [0.0, 0.9, 0.9, 0.0, 0.0],
        ]
        Sx, Sy = 2, 2
        d = split_labels_into_bins(labels, Sx, Sy)
        self.assertEqual(d[0, 0], [[0.0, 0.1, 0.1, 0.0, 0.0]])
        self.assertEqual(d[1, 0], [[0.0, 0.9, 0.1, 0.0, 0.0]])
        self.assertEqual(d[0, 1], [[0.0, 0.1, 0.9, 0.0, 0.0]])
        self.assertEqual(d[1, 1], [[0.0, 0.9, 0.9, 0.0, 0.0]])

    def test_labels_into_bins_3(self):
        sq0 = [
            [0.0, 0.1, 0.1, 0.0, 0.0],
            [0.0, 0.2, 0.2, 0.0, 0.0],
        ]
        sq1 = [
            [0.0, 0.8, 0.2, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0, 0.0],
        ]
        sq2 = [
            [0.0, 0.1, 0.9, 0.0, 0.0],
            [0.0, 0.2, 0.8, 0.0, 0.0],
        ]
        sq3 = [
            [0.0, 0.8, 0.8, 0.0, 0.0],
            [0.0, 0.9, 0.9, 0.0, 0.0],
        ]
        Sx, Sy, labels = 2, 2, [el for sq in [sq0, sq1, sq2, sq3] for el in sq]
        d = split_labels_into_bins(labels, Sx, Sy)
        for el in d[0, 0]:
            self.assertIn(el, sq0)
        for el in d[1, 0]:
            self.assertIn(el, sq1)
        for el in d[0, 1]:
            self.assertIn(el, sq2)
        for el in d[1, 1]:
            self.assertIn(el, sq3)

    def test_labels_into_bins_empty(self):
        Sx, Sy, labels = 2, 2, []
        d = split_labels_into_bins(labels, Sx, Sy)
        for coord, sorted_labels in d.items():
            self.assertEqual(sorted_labels, [])


if __name__ == "__main__":
    unittest.main()
