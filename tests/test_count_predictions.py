import torch
import unittest

from yogo.infer import count_cells_for_formatted_preds


class TestCountClassPredictions(unittest.TestCase):
    def test_simple_class_predictions(self):
        inp = torch.zeros(3, 5)
        inp[:, 0] = 1
        expected_result = torch.tensor([3, 0, 0, 0, 0], dtype=torch.long)
        torch.testing.assert_close(
            count_cells_for_formatted_preds(inp), expected_result
        )

    def test_float_predictions(self):
        row = torch.tensor([0.1, 0.2, 0.3, 0.4])
        inp = torch.stack([row, row, row])
        expected_result = torch.tensor([0, 0, 0, 3], dtype=torch.long)
        torch.testing.assert_close(
            count_cells_for_formatted_preds(inp), expected_result
        )

    def test_maked_predictions_filter_no_results(self):
        inp = torch.tensor(
            [[0.2, 0.4, 0.2, 0.2], [0.2, 0.4, 0.2, 0.2], [0.2, 0.4, 0.2, 0.2]]
        )
        expected_result = torch.tensor([0, 0, 0, 0], dtype=torch.long)
        torch.testing.assert_close(
            count_cells_for_formatted_preds(inp, min_confidence_threshold=0.6),
            expected_result,
        )

    def test_maked_predictions_filter(self):
        inp = torch.tensor(
            [[0.2, 0.7, 0.2, 0.2], [0.2, 0.4, 0.2, 0.2], [0.2, 0.4, 0.9, 0.2]]
        )
        expected_result = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        torch.testing.assert_close(
            count_cells_for_formatted_preds(inp, min_confidence_threshold=0.6),
            expected_result,
        )


if __name__ == "__main__":
    unittest.main()
