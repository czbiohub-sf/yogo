import torch
import unittest

from yogo.utils import format_preds


# TODO convert unittest to pytest
class TestFormatPreds(unittest.TestCase):
    def setUp(self):
        """
        build a tensor of shape (12,4,4)
        - 4x4 grid of predictions
        - first 4 elements of the prediction tensor is box coords in xcycwh format
        - 5th element is "objectness"
        - 6th-12th elements are class probabilities
        """
        # no predictions
        self.no_predictions = torch.zeros(12, 4, 4).float()

        # single prediction with box of 0 width and height
        self.single_prediction = torch.zeros(12, 4, 4).float()
        # setting objectness > thresh (default 0.5) in grid cell 0,0
        self.single_prediction[4, 0, 0] = 1.0
        # one-hot class pred
        self.single_prediction[5, :, :] = 1.0

        # single prediction with box of 0 width and height in xcycwh format
        # put it in grid cell 1,1 so we can combine with other predictions
        self.single_prediction_with_cxcywh_box = torch.zeros(12, 4, 4).float()
        self.single_prediction_with_cxcywh_box[5, :, :] = 1.0  # one-hot class pred
        self.single_prediction_with_cxcywh_box[4, 1, 1] = 1.0
        # box of width and height 0.1 at center (0.5,0.5) of image
        self.single_prediction_with_cxcywh_box[0, 1, 1] = 0.5
        self.single_prediction_with_cxcywh_box[1, 1, 1] = 0.5
        self.single_prediction_with_cxcywh_box[2, 1, 1] = 0.1
        self.single_prediction_with_cxcywh_box[3, 1, 1] = 0.1

    def test_no_predictions(self):
        """
        test that a tensor with no predictions returns an empty tensor of shape 0,12
        """
        torch.testing.assert_close(
            format_preds(self.no_predictions), torch.empty(0, 12)
        )

    def test_single_prediction(self):
        pred = format_preds(self.single_prediction)
        # should just filter pred tensor for the single prediction
        torch.testing.assert_close(pred, self.single_prediction[:, 0, 0].unsqueeze(0))

    def test_single_prediction_with_cxcywh_box(self):
        pred = format_preds(self.single_prediction_with_cxcywh_box)
        # make sure we get the prediction that we set, and that the box format is untouched
        # since the default is cxcywh
        torch.testing.assert_close(
            pred, self.single_prediction_with_cxcywh_box[:, 1, 1].unsqueeze(0)
        )

    def test_single_prediction_with_xyxy_box(self):
        pred = format_preds(self.single_prediction_with_cxcywh_box, box_format="xyxy")
        # make sure we get the prediction that we set, and that the box format is untouched
        # since the default is cxcywh
        actual = self.single_prediction_with_cxcywh_box[:, 1, 1].unsqueeze(0)
        actual[:, 0] = actual[:, 0] - actual[:, 2] / 2
        actual[:, 1] = actual[:, 1] - actual[:, 3] / 2
        actual[:, 2] = actual[:, 0] + actual[:, 2]
        actual[:, 3] = actual[:, 1] + actual[:, 3]
        torch.testing.assert_close(pred, actual)
