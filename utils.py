#! /usr/bin/env python3

import torch
import torchvision.transforms as T

from PIL import Image, ImageDraw
from typing import Optional, Union, List

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from typing import List, Dict


def format_preds_for_torchmetrics(batch_preds) -> List[Dict[str, torch.Tensor]]:
    batch_size, pred_shape, Sy, Sx = batch_preds.shape
    return [
        {
            "boxes": img_preds[:4, ...].view(4, Sy * Sx).T,
            "scores": img_preds[4, ...].view(Sy * Sx),
            "labels": torch.argmax(img_preds[5:, ...], dim=0).view(Sy * Sx),
        }
        for img_preds in batch_preds
    ]

def format_labels_for_torchmetrics(batch_labels) -> List[Dict[str, torch.Tensor]]:
    batch_size, label_shape, Sy, Sx = batch_labels.shape
    l = []
    for b, img_labels in enumerate(batch_labels):
        if torch.all(img_labels[0, ...] == 0).item():
            # mask says there are no labels!
            l.append({"boxes": torch.tensor([[]]), "labels": torch.tensor([])})
        else:
            # view -> T keeps tensor as a view, and no copies?
            row_ordered_img_labels = img_labels.view(-1, Sy * Sx).T
            # if label[0] == 0, there is no box in cell Sx/Sy - mask those out
            masked = row_ordered_img_labels[row_ordered_img_labels[..., 0] == 1, ...]
            l.append({ "boxes": masked[:, 1:5], "labels": masked[:, 5], })
    return l


def batch_mAP(self, batch_preds, batch_labels):
    formatted_batch_preds = format_preds_for_torchmetrics(batch_preds)
    formatted_batch_labels = format_labels_for_torchmetrics(batch_labels)
    metric = MeanAveragePrecision(box_format='xywh')
    metric.update(formatted_batch_preds, formatted_batch_labels)
    return metric.compute()


def draw_rects(
    img: torch.Tensor, rects: Union[torch.Tensor, List], thresh: Optional[float] = None
) -> Image:
    """
    img is the torch tensor representing an image
    rects is either
        - a torch.tensor of shape (pred, Sy, Sx), where pred = (xc, yc, w, h, confidence, ...)
        - a list of (class, xc, yc, w, h)
    thresh is a threshold for confidence when rects is a torch.Tensor
    """
    assert (
        len(img.shape) == 2
    ), f"takes single grayscale image - should be 2d, got {img.shape}"
    h, w = img.shape

    if isinstance(rects, torch.Tensor):
        pred_dim, Sy, Sx = rects.shape
        if thresh is None:
            thresh = 0.0
        rects = [r for r in rects.reshape(pred_dim, Sx * Sy).T if r[4] > thresh]
    elif isinstance(rects, list):
        if thresh is not None:
            raise ValueError("threshold only valid for tensor (i.e. prediction) input")
        rects = [r[1:] for r in rects]

    formatted_rects = [
        [
            int(w * (r[0] - r[2] / 2)),
            int(h * (r[1] - r[3] / 2)),
            int(w * (r[0] + r[2] / 2)),
            int(h * (r[1] + r[3] / 2)),
        ]
        for r in rects
    ]

    image = T.ToPILImage()(img[None, ...])
    rgb = Image.new("RGB", image.size)
    rgb.paste(image)
    draw = ImageDraw.Draw(rgb)

    for r in formatted_rects:
        draw.rectangle(r, outline="red")

    return rgb


if __name__ == "__main__":
    from model import YOGO
    from yogo_loss import YOGOLoss
    from cluster_anchors import best_anchor, get_all_bounding_boxes
    from dataloader import get_dataloader, load_dataset_description

    _, __, label_path, ___ = load_dataset_description("healthy_cell_dataset.yml")
    anchor_w, anchor_h = best_anchor(
        get_all_bounding_boxes(str(label_path), center_box=True)
    )

    dataloaders = get_dataloader("healthy_cell_dataset.yml", 16)
    DL = dataloaders["val"]
    Y = YOGO(anchor_w, anchor_h)
    Y.eval()

    for img_batch, label_batch in DL:
        out = Y(img_batch)
        formatted_label_batch = YOGOLoss.format_label_batch(out, label_batch)
        format_preds_for_torchmetrics(out)
        format_labels_for_torchmetrics(formatted_label_batch)
        print("SUCCESS WOO")
