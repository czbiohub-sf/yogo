import unittest
import torch

from yogo.infer import _count_class_predictions


class TestCountClassPredictions(unittest.TestCase):
    def test_simple_class_predictions(self):
        inp = torch.zeros(3, 5)
        inp[:, 0] = 1
        expected_result = torch.tensor([3, 0, 0, 0, 0], dtype=torch.long)
        torch.testing.assert_close(_count_class_predictions(inp), expected_result)

    def test_float_predictions(self):
        row = torch.tensor([0.1, 0.2, 0.3, 0.4])
        inp = torch.stack([row, row, row])
        expected_result = torch.tensor([0, 0, 0, 3], dtype=torch.long)
        torch.testing.assert_close(_count_class_predictions(inp), expected_result)


if __name__ == "__main__":
    unittest.main()
