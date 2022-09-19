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


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
    label_smoothing: float = 0
) -> torch.Tensor:
    # Adapted from link below to use softmax instead of sigmoid
    # https://pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html
    p = torch.softmax(inputs, dim=1)

    p_shaped = torch.permute(p, (0, 2, 3, 1))
    targets_shaped = torch.permute(p, (0, 2, 3, 1))

    p_t = p_shaped * targets_shaped + (1 - p_shaped) * (1 - targets_shaped)

    ce_loss = F.cross_entropy(inputs, targets[:, 0, :, :], reduction="none", label_smoothing=label_smoothing)
    ce_loss = ce_loss.unsqueeze(-1)

    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets_shaped + (1 - alpha) * (1 - targets_shaped)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class YOGOLoss(torch.nn.modules.loss._Loss):
    __constants__ = ["coord_weight", "no_obj_weight"]
    coord_weight: float
    no_obj_weight: float

    def __init__(
        self,
        coord_weight: float = 5.0,
        no_obj_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.coord_weight = coord_weight
        self.no_obj_weight = no_obj_weight
        self.mse = torch.nn.MSELoss(reduction="none")
        self.device = "cpu"

    def to(self, device):
        self.device = device
        super().to(device, dtype=torch.float32)
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
            * self.mse(pred_batch[:, 4, :, :], torch.zeros_like(pred_batch[:, 4, :, :]))
        ).sum()

        # objectness loss when there is an obj
        loss += (
            label_batch[:, 0, :, :]
            * self.mse(pred_batch[:, 4, :, :], torch.ones_like(pred_batch[:, 4, :, :]))
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
                    torch.sqrt(label_batch[:, 4, :, :]),
                )
            )
        ).sum()

        # classification loss
        loss += (
            torch.permute(label_batch, (0,2,3,1))[:, :, :, 0:1]
            * focal_loss(
                pred_batch[:, 5:, :, :],
                label_batch[:, 5:6, :, :].long(),
                reduction='none',
                label_smoothing=0.01
            )
        ).sum()

        return loss / batch_size

    @classmethod
    def format_labels(
        cls,
        pred_batch: torch.Tensor,
        label_batch: List[torch.Tensor],
        device: Union[str, torch.device] = "cpu",
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
            output = torch.zeros(batch_size, 1 + 4 + 1, Sy, Sx, device=device)
            for i, label_layer in enumerate(label_batch):
                label_cells = split_labels_into_bins(label_layer, Sx, Sy)

                for (k, j), labels in label_cells.items():
                    if len(labels) > 0:
                        # select best label by best IOU!
                        IoU = ops.box_iou(
                            ops.box_convert(
                                pred_batch[i, :4, j, k].unsqueeze(0), "cxcywh", "xyxy"
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


if __name__ == "__main__":
    from itertools import cycle
    from model import YOGO
    from dataloader import get_dataloader

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    def plot(data, l):
        "quick and hacky way to debug some split label jazz"
        LS = split_labels_into_bins(l, 9, 12)
        _, ax = plt.subplots()
        ax.imshow(data[0, 0, :, :])

        _, _, img_h, img_w = data.shape
        colors = cycle(["r", "g", "b", "c", "m", "y", "k"])
        # AAAHHHH! AHHHHHHH!!!!
        for k, ls in LS.items():
            current_axis = plt.gca()
            color = next(colors)
            for i, box in enumerate(ls):
                [_, xc, yc, w, h] = box
                current_axis.add_patch(
                    Rectangle(
                        (img_w * (xc - w / 2), img_h * (yc - h / 2)),
                        img_w * w,
                        img_h * h,
                        facecolor="none",
                        edgecolor=color,
                    )
                )
        plt.show()

    loss = YOGOLoss()

    m = YOGO(anchor_w=36, anchor_h=36)

    ODL = get_dataloader("healthy_cell_dataset.yml", batch_size=1)

    ii = 5
    for data, labels in ODL["test"]:
        for l in labels:
            plot(data, l)
            print(m(data).shape)

        ii -= 1
        if ii < 0:
            break
