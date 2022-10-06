"""
Implement YOLO loss function here
"""

import torch

import torch.nn.functional as F
import torchvision.ops as ops

from collections import defaultdict
from typing import Any, List, Dict, Tuple, Union

"""
Original YOLO paper did not mention IOU?
IOU Loss?
"""


class YOGOLoss(torch.nn.modules.loss._Loss):
    __constants__ = ["coord_weight", "no_obj_weight", "num_classes"]
    coord_weight: float
    no_obj_weight: float
    num_classes: int

    def __init__(
        self,
        coord_weight: float = 5.0,
        no_obj_weight: float = 0.5,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.coord_weight = coord_weight
        self.no_obj_weight = no_obj_weight
        self.num_classes = num_classes
        self.mse = torch.nn.MSELoss(reduction="none")
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

        TODO:
            - "bag of freebies" for free +5% mAP https://arxiv.org/pdf/1902.04103.pdf
            - Sigmoid + L2 loss -> BCE with logits for "Objectness"
            - See YoloV3 Sec 2.1

        """
        batch_size, preds_size, Sy, Sx = pred_batch.shape
        assert batch_size == len(label_batch)

        loss = torch.tensor(0, dtype=torch.float32, device=self.device)

        # objectness loss when there is no obj
        loss += (
            (1 - label_batch[:, 0, :, :])
            * self.no_obj_weight
            * self.mse(
                pred_batch[:, self.num_classes, :, :],
                torch.zeros_like(pred_batch[:, self.num_classes, :, :]),
            )
        ).sum()

        # objectness loss when there is an obj
        loss += (
            label_batch[:, 0, :, :]
            * self.mse(
                pred_batch[:, self.num_classes, :, :],
                torch.ones_like(pred_batch[:, self.num_classes, :, :]),
            )
        ).sum()

        # localization (i.e. xc, yc, w, h) loss
        loss += (
            label_batch[:, 0, :, :]
            * self.coord_weight
            * (
                self.mse(
                    pred_batch[:, 0, :, :],
                    label_batch[:, 1, :, :],
                )
                + self.mse(
                    pred_batch[:, 1, :, :],
                    label_batch[:, 2, :, :],
                )
                + self.mse(
                    torch.sqrt(pred_batch[:, 2, :, :]),
                    torch.sqrt(label_batch[:, 3, :, :]),
                )
                + self.mse(
                    torch.sqrt(pred_batch[:, 3, :, :]),
                    torch.sqrt(label_batch[:, self.num_classes, :, :]),
                )
            )
        ).sum()

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
        device: Union[str, torch.device] = "cpu",
        num_classes: int = 4,
    ) -> torch.Tensor:
        """
        input:
            pred_batch: shape (batch_size, preds_size, Sy, Sx)
            label_batch: List[torch.Tensor], and len(label_batch) == batch_size
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
                                pred_batch[i, :num_classes, j, k].unsqueeze(0),
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
