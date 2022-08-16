#! /usr/bin/env python3

import unittest
import numpy as np

from cluster_anchors import Box, area, iou, xc_yc_w_h_to_corners, corners_to_xc_yc_w_h


class TestClustering(unittest.TestCase):
    """
    it is best to draw out the bounding boxes for these tests
    """

    def test_IOU_sanity_checks(self):
        b1 = np.array([0, 1, 0, 1])
        b2 = np.array([1, 2, 1, 2])
        b3 = np.array([10, 11, 10, 11])
        self.assertEqual(iou(b1, b1), 1.0)
        self.assertEqual(iou(b2, b2), 1.0)
        self.assertEqual(iou(b1, b2), 0)
        self.assertEqual(iou(b1, b3), 0)

    def test_IOU_basic(self):
        b1 = np.array([0, 4, 2, 6])
        b2 = np.array([2, 6, 0, 4])
        self.assertEqual(iou(b1, b2), 4 / (4 * 4 + 4 * 4 - 4))
        self.assertEqual(iou(b2, b1), 4 / (4 * 4 + 4 * 4 - 4))

    def test_box_definition_conversions(self):
        for _ in range(100):
            corners = np.random.rand(4)
            self.assertTrue(
                np.allclose(
                    corners, xc_yc_w_h_to_corners(*corners_to_xc_yc_w_h(*corners))
                )
            )


if __name__ == "__main__":
    unittest.main()
