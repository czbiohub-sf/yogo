#! /usr/bin/env python3

import math

import torch
import torchvision.transforms as T

from PIL import Image, ImageDraw
from typing import Optional, Union, List

from torchmetrics import ConfusionMatrix
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from typing import Tuple, List, Dict


class Metrics:
    def __init__(self, num_classes=4, device="cpu", class_names=None):
        self.mAP = MeanAveragePrecision(box_format="cxcywh")
        self.confusion = ConfusionMatrix(num_classes=num_classes)
        self.confusion.to(device)

        self.class_names = (
            list(range(num_classes)) if class_names is None else class_names
        )

    def update(self, preds, labels, raw_preds=True):
        bs, pred_shape, Sy, Sx = preds.shape
        bs, label_shape, Sy, Sx = labels.shape

        mAP_preds, mAP_labels = self.format_for_mAP(preds, labels)

        confusion_preds, confusion_labels = self.format_for_confusion(
            preds, labels, raw_preds=raw_preds
        )

        self.mAP.update(mAP_preds, mAP_labels)
        self.confusion.update(confusion_preds, confusion_labels)

    def compute(self):
        confusion_mat = self.confusion.compute()

        nc1, nc2 = confusion_mat.shape
        assert nc1 == nc2

        L = []
        for i in range(nc1):
            for j in range(nc2):
                L.append(
                    (self.class_names[i], self.class_names[j], confusion_mat[i, j])
                )

        return self.mAP.compute(), L

    def reset(self):
        self.mAP.reset()
        self.confusion.reset()

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
            if torch.eq(torch.all(img_labels[0, ...] == 0), torch.tensor(False)):
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


def tupleize(v: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(v, int):
        return (v, v)
    return v


def convolution_output_shape(H_in, W_in, kernel_size, padding=0, stride=1, dilation=1):
    """
    simple calculator for output shape given input shape and conv2d params
    works for maxpool2d too

    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    kernel_size = tupleize(kernel_size)
    padding = tupleize(padding)
    stride = tupleize(stride)
    dilation = tupleize(dilation)

    H_out = math.floor(
        1 + (H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]
    )
    W_out = math.floor(
        1 + (W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]
    )

    print(H_out, W_out)
    return H_out, W_out


if __name__ == "__main__":
    x = 300, 300

    # backbone
    x = convolution_output_shape(*x, 3, padding=1)  # conv1
    x = convolution_output_shape(*x, 2, stride=2)  # maxp1
    print("block 1 done")

    x = convolution_output_shape(*x, 3, padding=1)  # conv1
    x = convolution_output_shape(*x, 2, stride=2)  # maxp1
    print("block 2 done")

    x = convolution_output_shape(*x, 3, padding=1)  # conv1
    x = convolution_output_shape(*x, 2, stride=2)  # maxp1
    print("block 3 done")

    x = convolution_output_shape(*x, 3, padding=1)  # conv1
    x = convolution_output_shape(*x, 2, stride=2)  # maxp1
    print("block 4 done")

    x = convolution_output_shape(*x, 3, padding=1)  # conv1
    print("block 5 done")

    # head
    x = convolution_output_shape(*x, 3, padding=1)  # conv1
    x = convolution_output_shape(
        *x,
        3,
    )  # conv1
    print("block 6 done")

    print(x)
