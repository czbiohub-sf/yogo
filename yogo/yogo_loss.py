import torch

from typing import Dict, Tuple

import torchvision.ops as ops


class YOGOLoss(torch.nn.modules.loss._Loss):
    __constants__ = ["no_obj_weight", "iou_weight", "classify_weight"]
    coord_weight: float
    no_obj_weight: float

    def __init__(
        self,
        no_obj_weight: float = 0.5,
        iou_weight: float = 5.0,
        classify_weight: float = 1.0,
        label_smoothing: float = 0.01,
    ) -> None:
        super().__init__()

        self.no_obj_weight = no_obj_weight
        self.iou_weight = iou_weight
        self.classify_weight = classify_weight

        self.mse = torch.nn.MSELoss(reduction="none")
        self.cel = torch.nn.CrossEntropyLoss(
            reduction="none", label_smoothing=label_smoothing
        )

        self.device = "cpu"

    def to(self, device):
        self.device = device
        super().to(device, non_blocking=True, dtype=torch.float32)
        return self

    def forward(
        self, pred_batch: torch.Tensor, label_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        pred and label are both 4d. pred_batch has shape
        (
             batch size,
             pred_dim,      (tx, ty, tw, th, to, c1, c2, c3, c4, ...)
             Sx,
             Sy
        )
        We return both the total loss (which can be backpropped) and individual
        loss components, which are useful for debugging.
        """
        batch_size, _, Sy, Sx = pred_batch.shape

        loss = torch.tensor(0, dtype=torch.float32, device=self.device)

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

        formatted_preds_masked = formatted_preds[:, mask].permute((1, 0))
        formatted_labels_masked = formatted_labels[:, mask].permute((1, 0))

        formatted_preds_xyxy = ops.box_convert(
            formatted_preds_masked,
            "cxcywh",
            "xyxy",
        )

        valid_box_mask = torch.logical_and(
            formatted_preds_xyxy[:, 0] != formatted_preds_xyxy[:, 2],
            formatted_preds_xyxy[:, 1] != formatted_preds_xyxy[:, 3],
        )

        formatted_preds_xyxy = formatted_preds_xyxy[valid_box_mask]
        formatted_labels_masked = formatted_labels_masked[valid_box_mask]

        iou_loss = (
            self.iou_weight
            * (
                ops.complete_box_iou_loss(
                    torch.clamp(
                        formatted_preds_xyxy,
                        min=0,
                        max=1,
                    ),
                    formatted_labels_masked,
                )
            ).sum()
            / batch_size
        )

        classification_loss = (
            self.classify_weight
            * (
                label_batch[:, 0, :, :]
                * self.cel(pred_batch[:, 5:, :, :], label_batch[:, 5, :, :].long())
            ).sum()
            / batch_size
        )

        objectness_loss = (
            self.mse(pred_batch[:, 4, :, :], label_batch[:, 0, :, :])
            * (label_batch[:, 0, :, :] * (1 - self.no_obj_weight) + self.no_obj_weight)
        ).sum() / batch_size

        loss = objectness_loss + iou_loss + classification_loss

        loss_components = {
            "iou_loss": iou_loss.item(),
            "objectness_loss": objectness_loss.item(),
            "classification_loss": classification_loss.item(),
        }

        return loss, loss_components
