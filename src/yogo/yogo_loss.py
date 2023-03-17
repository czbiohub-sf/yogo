import torch

import torchvision.ops as ops


class YOGOLoss(torch.nn.modules.loss._Loss):
    __constants__ = ["coord_weight", "no_obj_weight", "num_classes"]
    coord_weight: float
    no_obj_weight: float
    num_classes: int

    # TODO sweep over coord + no_obj_weight, look at confusion matrix for results
    def __init__(
        self,
        coord_weight: float = 5.0,
        no_obj_weight: float = 0.5,
        classify: bool = True,
    ) -> None:
        super().__init__()
        self.coord_weight = coord_weight
        self.no_obj_weight = no_obj_weight
        self.bce = torch.nn.BCELoss(reduction="none")
        self._classify = classify

        # TODO sweep over label_smoothing values
        if self._classify:
            self.cel = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=0.01)

        self.device = "cpu"

    def to(self, device):
        self.device = device
        super().to(device, non_blocking=True, dtype=torch.float32)
        return self

    def forward(
        self, pred_batch: torch.Tensor, label_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        pred and label are both 4d. pred_batch has shape
        (
             batch size,
             pred_dim,      (tx, ty, tw, th, to, c1, c2, c3, c4)
             Sx,
             Sy
        )
        """
        batch_size, _, Sy, Sx = pred_batch.shape

        loss = torch.tensor(0, dtype=torch.float32, device=self.device)

        # objectness loss when there is no obj
        loss += (
            self.no_obj_weight
            * (
                (1 - label_batch[:, 0, :, :])
                * self.bce(
                    pred_batch[:, 4, :, :],
                    torch.zeros_like(pred_batch[:, 4, :, :]),
                )
            ).sum()
        )

        # objectness loss when there is an obj
        loss += (
            label_batch[:, 0, :, :]
            * self.bce(
                pred_batch[:, 4, :, :],
                torch.ones_like(pred_batch[:, 4, :, :]),
            )
        ).sum()

        # bounding box loss
        # there is a lot of work to get it into the right format for loss
        # hopefully it is not too slow
        formatted_preds = (
            pred_batch[:, :4, :, :]
            .permute((1, 0, 2, 3))
            .reshape(4, batch_size * Sx * Sy)
        )
        formatted_labels = (
            label_batch[:, 1:5, :, :]
            .permute((1, 0, 2, 3))
            .reshape(4, batch_size * Sx * Sy)
        )
        mask = (
            label_batch[:, 0:1, :, :]
            .permute((1, 0, 2, 3))
            .reshape(batch_size * Sx * Sy)
        ).bool()

        # TODO try .T
        formatted_preds_masked = formatted_preds[:, mask].permute((1, 0))
        formatted_labels_masked = formatted_labels[:, mask].permute((1, 0))

        loss += (
            self.coord_weight
            * (
                ops.complete_box_iou_loss(
                    torch.clamp(
                        ops.box_convert(
                            formatted_preds_masked,
                            "cxcywh",
                            "xyxy",
                        ),
                        min=0,
                        max=1,
                    ),
                    formatted_labels_masked,
                )
            ).sum()
        )

        # classification loss
        if self._classify:
            loss += (
                label_batch[:, 0, :, :]
                * self.cel(pred_batch[:, 5:, :, :], label_batch[:, 5, :, :].long())
            ).sum()

        return loss / batch_size
