#! /usr/bin/env python3

import torch
import torchvision.transforms as T

from PIL import Image, ImageDraw
from typing import Optional, Union, List

from torchmetrics import ConfusionMatrix
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from typing import Optional, Tuple, List, Dict


class Metrics:
    # TODO num classes?
    def __init__(self, num_classes: int, device: str="cpu", class_names: Optional[List[str]]=None):
        self.mAP = MeanAveragePrecision(box_format="cxcywh")
        #self.confusion = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        #self.confusion.to(device)

        self.class_names = (
            list(range(num_classes)) if class_names is None else class_names
        )

    def update(self, preds, labels, raw_preds=True):
        bs, pred_shape, Sy, Sx = preds.shape
        bs, label_shape, Sy, Sx = labels.shape

        mAP_preds, mAP_labels = self.format_for_mAP(preds, labels)
        self.mAP.update(mAP_preds, mAP_labels)

        # confusion_preds, confusion_labels = self.format_for_confusion(
        #    preds, labels, raw_preds=raw_preds
        #)
        #self.confusion.update(confusion_preds, confusion_labels)

    def compute(self):
        # confusion_mat = self.confusion.compute()

        # nc1, nc2 = confusion_mat.shape
        # assert nc1 == nc2

        L = []
        """
        for i in range(nc1):
            for j in range(nc2):
                L.append(
                    (self.class_names[i], self.class_names[j], confusion_mat[i, j])
                )
        """

        return self.mAP.compute(), L

    def reset(self):
        self.mAP.reset()
        #self.confusion.reset()

    @staticmethod
    def format_for_confusion(
        batch_preds, batch_labels, raw_preds=True
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        bs, pred_shape, Sy, Sx = batch_preds.shape
        bs, label_shape, Sy, Sx = batch_labels.shape

        if raw_preds:
            batch_preds[:, 5:, :, :] = torch.softmax(batch_preds[:, 5:, :, :], dim=1)

        confusion_batch_preds = (
            batch_preds.permute(1, 0, 2, 3)[5:, ...].reshape(-1, bs * Sx * Sy).T
        )
        confusion_labels = (
            batch_labels.permute(1, 0, 2, 3)[5, :, :, :]
            .reshape(1, bs * Sx * Sy)
            .permute(1, 0)
            .long()
        )
        return confusion_batch_preds, confusion_labels

    @staticmethod
    def format_for_mAP(
        batch_preds, batch_labels
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        bs, label_shape, Sy, Sx = batch_labels.shape
        bs, pred_shape, Sy, Sx = batch_preds.shape

        device = batch_preds.device
        preds, labels = [], []
        for b, (img_preds, img_labels) in enumerate(zip(batch_preds, batch_labels)):
            if torch.all(img_labels[0, ...] == 0).item():
                # mask says there are no labels!
                labels.append(
                    {
                        "boxes": torch.tensor([], device=device),
                        "labels": torch.tensor([], device=device),
                    }
                )
                preds.append(
                    {
                        "boxes": torch.tensor([], device=device),
                        "labels": torch.tensor([], device=device),
                        "scores": torch.tensor([], device=device),
                    }
                )
            else:
                # view -> T keeps tensor as a view, and no copies?
                row_ordered_img_preds = img_preds.view(-1, Sy * Sx).T
                row_ordered_img_labels = img_labels.view(-1, Sy * Sx).T

                # if label[0] == 0, there is no box in cell Sx/Sy - mask those out
                mask = row_ordered_img_labels[..., 0] == 1

                labels.append(
                    {
                        "boxes": row_ordered_img_labels[mask, 1:5],
                        "labels": row_ordered_img_labels[mask, 5],
                    }
                )
                preds.append(
                    {
                        "boxes": row_ordered_img_preds[mask, :4],
                        "scores": row_ordered_img_preds[mask, 4],
                        "labels": torch.argmax(row_ordered_img_preds[mask, 5:], dim=1),
                    }
                )

        return preds, labels


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
        formatted_rects = [
            [
                int(w * (r[0] - r[2] / 2)),
                int(h * (r[1] - r[3] / 2)),
                int(w * (r[0] + r[2] / 2)),
                int(h * (r[1] + r[3] / 2)),
                torch.argmax(r[5:]).item(),
            ]
            for r in rects
        ]
    elif isinstance(rects, list):
        if thresh is not None:
            raise ValueError("threshold only valid for tensor (i.e. prediction) input")
        formatted_rects = [
            [
                int(w * (r[1] - r[3] / 2)),
                int(h * (r[2] - r[4] / 2)),
                int(w * (r[1] + r[3] / 2)),
                int(h * (r[2] + r[4] / 2)),
                r[0],
            ]
            for r in rects
        ]

    image = T.ToPILImage()(img[None, ...])
    rgb = Image.new("RGB", image.size)
    rgb.paste(image)
    draw = ImageDraw.Draw(rgb)

    for r in formatted_rects:
        draw.rectangle(r[:4], outline="red")
        draw.text((r[0], r[1]), str(r[4]), (0, 0, 0))

    return rgb


if __name__ == "__main__":
    import sys

    from matplotlib.pyplot import imshow, show
    from pathlib import Path

    from yogo.dataloader import get_dataloader
    from yogo.data_transforms import RandomVerticalCrop

    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path to image or dir of images>")
        sys.exit(1)

    path_to_ddf = sys.argv[1]
    ds = get_dataloader(
        path_to_ddf,
        batch_size=1,
        training= False,
    )

    for img, label in ds["val"]:
        imshow(draw_rects(img[0,0,...], list(label[0])))
        show()
