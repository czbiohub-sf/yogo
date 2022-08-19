"""
Implement YOLO loss function here
"""

import torch


class YOGOLoss(torch.nn.modules.loss._Loss):
    __constants__ = ["cord_weight", "no_obj_weight", "reduction"]
    cord_weight: float
    no_obj_weight: float

    def __init__(
        self,
        cord_weight: float = 5.0,
        no_obj_weight: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.cord_weight = cord_weight
        self.no_obj_weight = no_obj_weight
        self.mse = torch.nn.MSELoss()

    def forward(self, y_pred, y_true) -> torch.Tensor:
        """
        y_pred and y_true has shape
        (
             batch size,
             # anchors,
             pred_dim (5 + #classes),
             Sx,
             Sy
        )

        TODO: impl kroneker-delta for
            - object in cell
            - whether the preictor is "responsible" for that prediction
        """

        pass
