#! /usr/bin/env python3

import torch
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

    def test_IOU_sanity_checks_tensor(self) -> None:
        b1 = torch.tensor([0, 1, 0, 1])
        b2 = torch.tensor([1, 2, 1, 2])
        b3 = torch.tensor([10, 11, 10, 11])
        self.assertEqual(torch_iou(b1, b1), torch.tensor(1.0))
        self.assertEqual(torch_iou(b2, b2), torch.tensor(1.0))
        self.assertEqual(torch_iou(b1, b2), torch.tensor(0))
        self.assertEqual(torch_iou(b1, b3), torch.tensor(0))

    def test_IOU_basic_tensor(self) -> None:
        b1 = torch.tensor([0, 4, 2, 6])
        b2 = torch.tensor([2, 6, 0, 4])
        self.assertEqual(torch_iou(b1, b2), torch.tensor(4 / (4 * 4 + 4 * 4 - 4)))
        self.assertEqual(torch_iou(b2, b1), torch.tensor(4 / (4 * 4 + 4 * 4 - 4)))

    def test_box_definition_conversions(self) -> None:
        for i in range(100):
            corners = np.random.rand(6, 4)
            self.assertTrue(
                np.allclose(
                    corners, xc_yc_w_h_to_corners(corners_to_xc_yc_w_h(corners))
                )
            )

    def test_box_torch_definition_conversion(self) -> None:
        for i in range(100):
            corners = torch.rand(6, 4)
            self.assertTrue(
                np.allclose(
                    corners, xc_yc_w_h_to_corners(corners_to_xc_yc_w_h(corners))
                )
            )


class TestLossUtilities(unittest.TestCase):
    def assertTensorEq(self, a, b):
        self.assertTrue(torch.equal(a, b))

    def test_labels_into_bins_1(self):
        labels = torch.tensor(
            [
                [0.0, 0.1, 0.1, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0, 0.0],
            ]
        )
        Sx, Sy = 2, 1
        d = split_labels_into_bins(labels, Sx, Sy)
        self.assertTensorEq(d[0], torch.tensor([0, 0]))
        self.assertTensorEq(d[1], torch.tensor([1, 0]))

    def test_labels_into_bins_2(self):
        Sx, Sy, labels = (
            2,
            2,
            torch.tensor(
                [
                    [0.0, 0.1, 0.1, 0.0, 0.0],
                    [0.0, 0.9, 0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.9, 0.0, 0.0],
                    [0.0, 0.9, 0.9, 0.0, 0.0],
                ]
            ),
        )
        d = split_labels_into_bins(labels, Sx, Sy)
        self.assertTensorEq(d[0], torch.tensor([0, 0]))
        self.assertTensorEq(d[1], torch.tensor([1, 0]))
        self.assertTensorEq(d[2], torch.tensor([0, 1]))
        self.assertTensorEq(d[3], torch.tensor([1, 1]))

    def test_labels_into_bins_3(self):
        labels = torch.tensor(
            [
                [0.0, 0.1, 0.1, 0.0, 0.0],
                [0.0, 0.2, 0.2, 0.0, 0.0],
                [0.0, 0.8, 0.2, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0, 0.0],
                [0.0, 0.1, 0.9, 0.0, 0.0],
                [0.0, 0.2, 0.8, 0.0, 0.0],
                [0.0, 0.8, 0.8, 0.0, 0.0],
                [0.0, 0.9, 0.9, 0.0, 0.0],
            ]
        )
        Sx, Sy = 2, 2
        self.assertTensorEq(
            split_labels_into_bins(labels, Sx, Sy),
            torch.tensor(
                [
                    [0, 0],
                    [0, 0],
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [1, 1],
                    [1, 1],
                ]
            ),
        )

    def test_labels_into_bins_tensor(self):
        label_batch = torch.tensor(
            [
                [
                    [0.0, 0.1, 0.1, 0.0, 0.0],
                    [0.0, 0.9, 0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.9, 0.0, 0.0],
                    [0.0, 0.9, 0.9, 0.0, 0.0],
                ],
                [
                    [0.0, 0.1, 0.1, 0.0, 0.0],
                    [0.0, 0.9, 0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.9, 0.0, 0.0],
                    [0.0, 0.9, 0.9, 0.0, 0.0],
                ],
            ]
        )
        d = split_labels_into_bins(label_batch, 2, 2)
        self.assertTensorEq(d[0, 0], torch.tensor([0, 0]))
        self.assertTensorEq(d[0, 1], torch.tensor([1, 0]))
        self.assertTensorEq(d[0, 2], torch.tensor([0, 1]))
        self.assertTensorEq(d[0, 3], torch.tensor([1, 1]))
        self.assertTensorEq(d[1, 0], torch.tensor([0, 0]))
        self.assertTensorEq(d[1, 1], torch.tensor([1, 0]))
        self.assertTensorEq(d[1, 2], torch.tensor([0, 1]))
        self.assertTensorEq(d[1, 3], torch.tensor([1, 1]))


if __name__ == "__main__":
    unittest.main()
