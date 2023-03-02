import torch

import torch.nn.functional as F
import torchvision.ops as ops

from collections import defaultdict
from typing import Any, List, Dict, Tuple, Union


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
    ) -> None:
        super().__init__()
        self.coord_weight = coord_weight
        self.no_obj_weight = no_obj_weight
        self.mse = torch.nn.MSELoss(reduction="none")
        # TODO sweep over label_smoothing values
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
        batch_size, preds_size, Sy, Sx = pred_batch.shape

        loss = torch.tensor(0, dtype=torch.float32, device=self.device)

        # objectness loss when there is no obj
        loss += (
            self.no_obj_weight
            * (
                (1 - label_batch[:, 0, :, :])
                * self.mse(
                    pred_batch[:, 4, :, :],
                    torch.zeros_like(pred_batch[:, 4, :, :]),
                )
            ).sum()
        )

        # objectness loss when there is an obj
        loss += (
            label_batch[:, 0, :, :]
            * self.mse(
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
                    ops.box_convert(
                        formatted_labels_masked,
                        "cxcywh",
                        "xyxy",
                    ),
                )
            ).sum()
        )

        # classification loss
        loss += (
            label_batch[:, 0, :, :]
            * self.cel(pred_batch[:, 5:, :, :], label_batch[:, 5, :, :].long())
        ).sum()

        return loss / batch_size

    @classmethod
    def format_labels(
        cls,
        pred_batch: torch.Tensor,
        label_batch: List[torch.Tensor],
        num_classes: int,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        input:
            pred_batch: shape (batch_size, preds_size, Sy, Sx)
            label_batch: List[torch.Tensor], and len(label_batch) == batch_size
            num_classes: int
        output:
            torch.Tensor of shape (batch_size, masked_label_len, Sy, Sx)

        dimension masked_label is [mask, xc, yc, w, h, *classes], where mask == 1
        if there is a label associated with (Sy,Sx) at the given batch, else 0. If
        mask is 0, then the rest of the label values are "don't care" values (just
        setting to 0 is fine).

        TODO: maybe we can drop some sync points by converting label_batch to tensor?
        Have a parameter for "num labels" or smth, and have all tensors be the size
        of the minimum tensor size (instead of having a list)
        """
        batch_size, preds_size, Sy, Sx = pred_batch.shape
        with torch.no_grad():
            output = torch.zeros(batch_size, 1 + num_classes + 1, Sy, Sx, device=device)
            for i, label_layer in enumerate(label_batch):
                label_cells = split_labels_into_bins(label_layer, Sx, Sy)

                for (k, j), labels in label_cells.items():
                    if len(labels) > 0:
                        # select best label by best IOU!
                        IoU = ops.box_iou(
                            ops.box_convert(
                                pred_batch[i, :4, j, k].unsqueeze(0),
                                "cxcywh",
                                "xyxy",
                            ),
                            ops.box_convert(labels[:, 1:], "cxcywh", "xyxy"),
                        )
                        pred_square_idx = torch.argmax(IoU)
                        output[i, 0, j, k] = 1
                        output[i, 1:5, j, k] = labels[pred_square_idx][1:]
                        output[i, 5, j, k] = labels[pred_square_idx][0]

            return output


def split_labels_into_bins(
    labels: torch.Tensor, Sx, Sy
) -> Dict[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    # it is really a single-element long tensor
    d: Dict[Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]] = defaultdict(list)
    for label in labels:
        i = torch.div(label[1], (1 / Sx), rounding_mode="trunc").long()
        j = torch.div(label[2], (1 / Sy), rounding_mode="trunc").long()
        d[(i, j)].append(label)
    return {k: torch.vstack(vs) for k, vs in d.items()}
