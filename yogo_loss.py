"""
Implement YOLO loss function here
"""

import torch
import bisect

import torch.nn.functional as F

from operator import itemgetter
from typing import Any, List, Dict, Tuple, Callable
from collections import defaultdict

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
        self.cel = torch.nn.CrossEntropyLoss()
        self.device = "cpu"

    def to(self, device):
        # FIXME: hack?
        self.device = device
        super().to(device)
        return self

    def forward(
        self, pred_batch: torch.Tensor, label_batch: List[List[List[float]]]
    ) -> torch.Tensor:
        """
        pred and label are both 4d. pred_batch has shape
        (
             batch size,
             pred_dim, (tx, ty, tw, th, to, c1, c2, c3, c4)
             Sx,
             Sy
        )

        TODO:
            - "bag of freebies" for free +5% mAP https://arxiv.org/pdf/1902.04103.pdf
        """
        # TODO - this will be halariously slow! must speed up shortly
        # TODO - need the lables to be formatted correctly! See following link for loss impl.
        # https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
        # IOU may need to be rewritten - used in `t_o` preds
        batch_size, preds_size, Sy, Sx = pred_batch.shape
        assert batch_size == len(label_batch)

        loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        for i in range(batch_size):
            bins = split_labels_into_bins(label_batch[i], Sx, Sy)

            for (k, j), Ls in bins.items():
                if len(Ls) == 0:
                    objectness = self.no_obj_weight * self.mse(
                        pred_batch[i, 4, j, k], torch.tensor(0.0, device=self.device)
                    )
                    loss += objectness
                elif len(Ls) >= 1:
                    [cls, xc, yc, w, h] = Ls.pop()
                    localization = self.coord_weight * (
                        self.mse(
                            pred_batch[i, 0, j, k], torch.tensor(xc, device=self.device)
                        )
                        + self.mse(
                            pred_batch[i, 1, j, k], torch.tensor(yc, device=self.device)
                        )
                        + self.mse(
                            torch.sqrt(pred_batch[i, 2, j, k]),
                            torch.sqrt(torch.tensor(w, device=self.device)),
                        )
                        + self.mse(
                            torch.sqrt(pred_batch[i, 3, j, k]),
                            torch.sqrt(torch.tensor(h, device=self.device)),
                        )
                    )
                    objectness = self.mse(
                        pred_batch[i, 4, j, k], torch.tensor(1.0, device=self.device)
                    )
                    classification = self.cel(
                        pred_batch[i, 5:, j, k],
                        torch.tensor(int(cls), dtype=torch.long, device=self.device),
                    )
                    loss += localization
                    loss += objectness
                    loss += classification
                else:
                    # TODO: impl!
                    pass
        return loss / batch_size


def split_labels_into_bins(labels, Sx, Sy) -> Dict[Tuple[int, int], List[List[float]]]:
    d: Dict[Tuple[int, int], List[List[float]]] = defaultdict(list)
    for label in labels:
        d[(label[1] // (1 / Sx), label[2] // (1 / Sy))].append(label)
    return d


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
