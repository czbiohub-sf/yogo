"""
Implement YOLO loss function here
"""

import torch
import bisect

from operator import itemgetter
from typing import List, Any, Tuple, Callable

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
            - "bag of freebies" for free +5% mAP https://arxiv.org/pdf/1902.04103.pdf
        """
        # TODO - need the lables to be formatted correctly! See following link for details.
        # https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
        # IOU may need to be rewritten - used in `t_o` preds
        batch_size, preds_size, Sx, Sy = pred.shape
        assert batch_size == len(labels)

        # return self.coord_weight * (
        # return self.coord_weight * (
        #     self.mse(pred[:, :2, :, :], true[:, :2, :, :])
        #     + self.mse(torch.sqrt(pred[:, :2, :, :]), torch.sqrt(true[:, :2, :, :]))
        # )


def _split_labels_into_bins(labels, Sx, Sy):
    bins: Dict[Tuple[int, int], List[List[float]]] = dict()
    labels_x_sorted = sorted(labels, key=itemgetter(1))
    x_bins = _get_bin_sorted_list_idxs(labels_x_sorted, Sx, key=itemgetter(1))
    for i, x_bin in enumerate(x_bins):
        labels_slice_y_sorted = sorted(labels_x_sorted[x_bin], key=itemgetter(2))
        y_bins = _get_bin_sorted_list_idxs(labels_slice_y_sorted, Sy, key=itemgetter(2))
        for j, y_bin in enumerate(y_bins):
            bins[(i, j)] = labels_slice_y_sorted[y_bin]

    return bins


def _get_bin_sorted_list_idxs(
    arr: List[Any], max_num_bins: int, key: Callable = lambda a: a
) -> List[slice]:
    lo = 0
    l: List[slice] = []
    bin_width = 1 / max_num_bins
    keys = [key(v) for v in arr]
    for i in range(1, max_num_bins + 1):
        boundary = bisect.bisect(keys, i * bin_width, lo=lo)
        l.append(slice(lo, boundary))
        lo = boundary
    return l


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from itertools import cycle
    from cluster_anchors import plot_boxes
    from model import YOGO
    from dataloader import get_dataloader

    loss = YOGOLoss()

    m = YOGO(anchor_w=36, anchor_h=36)

    ODL = get_dataloader("healthy_cell_dataset.yml", batch_size=1)
    ii = 5
    for data, labels in ODL["test"]:
        for l in labels:
            LS = _split_labels_into_bins(l, 4, 4)
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
        ii -= 1
        if ii < 0:
            raise RuntimeError("im freaking out")
