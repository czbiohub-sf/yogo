#! /usr/bin/env python3

import unittest

from cluster_anchors import Box, area, iou


class TestClustering(unittest.TestCase):
    """
    it is best to draw out the bounding boxes for these tests
    """
    def test_IOU_sanity_checks(self):
        b1 = Box(xmin=0, ymin=0, xmax=1, ymax=1)
        b2 = Box(xmin=1, ymin=1, xmax=2, ymax=2)
        b3 = Box(xmin=10, ymin=10, xmax=11, ymax=11)
        self.assertEqual(iou(b1, b1), 1.0)
        self.assertEqual(iou(b2, b2), 1.0)
        self.assertEqual(iou(b1, b2), 0)
        self.assertEqual(iou(b1, b3), 0)

    def test_IOU_basic(self):
        b1 = Box(xmin=0, ymin=2, xmax=4, ymax=6)
        b2 = Box(xmin=2, ymin=0, xmax=6, ymax=4)
        self.assertEqual(iou(b1, b2), 4 / (4*4 + 4*4 - 4))
        self.assertEqual(iou(b2, b1), 4 / (4*4 + 4*4 - 4))


if __name__ == "__main__":
    unittest.main()
