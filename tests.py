#! /usr/bin/env python3

import torch
import unittest
import numpy as np

from cluster_anchors import *
from yogo_loss import split_labels_into_bins, split_labels_into_bins_tensor


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
        b3 = np.array([0, 1, 0, 1])
        b4 = np.array([0, 0.5, 0, 0.5])
        self.assertEqual(iou(b1, b2), 4 / (4 * 4 + 4 * 4 - 4))
        self.assertEqual(iou(b2, b1), 4 / (4 * 4 + 4 * 4 - 4))
        self.assertEqual(iou(b3, b4), (0.5 * 0.5) / (0.5 * 0.5 + 1 * 1 - 0.5 * 0.5))

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

    def test_finding_max_IOU_tensor(self) -> None:
        b1 = torch.tensor([0, 1, 0, 1])
        b2 = torch.tensor(
            [
                [0, 0.5, 0, 0.5],
                [0, 0.25, 0, 0.25],
                [0, 0.125, 0, 0.125],
            ]
        )
        out = torch_iou(b1, b2)
        self.assertEqual(out[0], 0.5**2 / (1**2 + 0.5**2 - 0.5**2))
        self.assertEqual(out[1], 0.25**2 / (1**2 + 0.25**2 - 0.25**2))
        self.assertEqual(out[2], 0.125**2 / (1**2 + 0.125**2 - 0.125**2))

    def test_box_definition_conversions(self) -> None:
        for i in range(100):
            corners = np.random.rand(6, 4)
            self.assertTrue(
                np.allclose(corners, centers_to_corners(corners_to_centers(corners)))
            )

    def test_box_torch_definition_conversion(self) -> None:
        for i in range(100):
            corners = torch.rand(6, 4)
            self.assertTrue(
                np.allclose(corners, centers_to_corners(corners_to_centers(corners)))
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
        self.assertTensorEq(d[0, 0], torch.tensor([[0.0, 0.1, 0.1, 0.0, 0.0]]))
        self.assertTensorEq(d[1, 0], torch.tensor([[0.0, 0.9, 0.1, 0.0, 0.0]]))

    def test_labels_into_bins_2(self):
        labels = torch.tensor(
            [
                [0.0, 0.1, 0.1, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0, 0.0],
                [0.0, 0.1, 0.9, 0.0, 0.0],
                [0.0, 0.9, 0.9, 0.0, 0.0],
            ]
        )
        Sx, Sy = 2, 2
        d = split_labels_into_bins(labels, Sx, Sy)
        self.assertTensorEq(d[0, 0], torch.tensor([[0.0, 0.1, 0.1, 0.0, 0.0]]))
        self.assertTensorEq(d[1, 0], torch.tensor([[0.0, 0.9, 0.1, 0.0, 0.0]]))
        self.assertTensorEq(d[0, 1], torch.tensor([[0.0, 0.1, 0.9, 0.0, 0.0]]))
        self.assertTensorEq(d[1, 1], torch.tensor([[0.0, 0.9, 0.9, 0.0, 0.0]]))

    def test_labels_into_bins_3(self):
        sq0 = torch.tensor(
            [
                [0.0, 0.1, 0.1, 0.0, 0.0],
                [0.0, 0.2, 0.2, 0.0, 0.0],
            ]
        )
        sq1 = torch.tensor(
            [
                [0.0, 0.8, 0.2, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0, 0.0],
            ]
        )
        sq2 = torch.tensor(
            [
                [0.0, 0.1, 0.9, 0.0, 0.0],
                [0.0, 0.2, 0.8, 0.0, 0.0],
            ]
        )
        sq3 = torch.tensor(
            [
                [0.0, 0.8, 0.8, 0.0, 0.0],
                [0.0, 0.9, 0.9, 0.0, 0.0],
            ]
        )
        Sx, Sy, labels = (
            2,
            2,
            torch.vstack([el for sq in [sq0, sq1, sq2, sq3] for el in sq]),
        )
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

    def test_labels_into_bins_tensor_1(self):
        labels = torch.tensor(
            [
                [0.0, 0.1, 0.1, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0, 0.0],
            ]
        )
        Sx, Sy = 2, 1
        d = split_labels_into_bins_tensor(labels, Sx, Sy)
        self.assertTensorEq(d[0], torch.tensor([0, 0]))
        self.assertTensorEq(d[1], torch.tensor([1, 0]))

    def test_labels_into_bins_tensor_2(self):
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
        d = split_labels_into_bins_tensor(labels, Sx, Sy)
        self.assertTensorEq(d[0], torch.tensor([0, 0]))
        self.assertTensorEq(d[1], torch.tensor([1, 0]))
        self.assertTensorEq(d[2], torch.tensor([0, 1]))
        self.assertTensorEq(d[3], torch.tensor([1, 1]))

    def test_labels_into_bins_tensor_3(self):
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
            split_labels_into_bins_tensor(labels, Sx, Sy),
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

    def test_labels_into_bins_tensors(self):
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
        d = split_labels_into_bins_tensor(label_batch, 2, 2)
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
