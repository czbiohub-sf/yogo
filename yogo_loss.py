"""
Implement YOLO loss function here
"""

import torch

from typing import List, Any

"""
Original YOLO paper did not mention IOU?
IOU Loss?
"""


class YOGOLoss(torch.nn.modules.loss._Loss):
    __constants__ = ["coord_weight", "no_obj_weight", "reduction"]
    coord_weight: float
    no_obj_weight: float

    def __init__(
        self,
        coord_weight: float = 5.0,
        no_obj_weight: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.coord_weight = coord_weight
        self.no_obj_weight = no_obj_weight
        self.mse = torch.nn.MSELoss()

    def forward(self, pred: torch.Tensor, labels: List[List[float]]) -> torch.Tensor:
        """
        pred and true has shape
        (
             batch size,
             pred_dim (5 + #classes), (tx, ty, tw, th, to, c1, c2, c3, c4)
             Sx,
             Sy
        )

        TODO:
            - impl kroneker-delta for
                - object in cell
                - whether the predctor is "responsible" for that prediction
            - class smoothing https://arxiv.org/pdf/1902.04103.pdf
        """
        # TODO - need the lables to be formatted correctly! See following link for details.
        # https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
        # IOU may need to be rewritten - used in to preds
        batch_size, preds_size, Sx, Sy = pred.shape
        return self.coord_weight * (
            self.mse(pred[:, :2, :, :], true[:, :2, :, :])
            + self.mse(torch.sqrt(pred[:, :2, :, :]), torch.sqrt(true[:, :2, :, :]))
        )
