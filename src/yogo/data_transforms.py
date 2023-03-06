import torch
import torchvision
import torchvision.transforms.functional as F

from typing import Sequence, Tuple, List, Any, cast


class DualInputModule(torch.nn.Module):
    def forward(self, inpt_a, inpt_b):
        ...


class DualInputId(DualInputModule):
    def __init__(self, *args: DualInputModule):
        super().__init__()

    def forward(self, img_batch, labels):
        return img_batch, labels


class MultiArgSequential(torch.nn.Sequential):
    def __init__(self, *args: DualInputModule, **kwargs):
        # Filter out Id transforms for mild performance gain
        super().__init__(*[t for t in args if not isinstance(t, DualInputId)], **kwargs)

    def forward(self, *input):
        for module in self:
            input = module(*input)
        return input


class ImageTransformLabelIdentity(DualInputModule):
    """
    A transform for images that leaves alone the labels,
    good for e.g. resizes
    """

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, img_batch, labels):
        return self.transform(img_batch), labels


class RandomVerticalCrop(DualInputModule):
    """Random crop of a horizontal strip over the batch."""

    def __init__(self, height: float):
        if not (0 < height < 1):
            raise ValueError(
                f"height for RandomCrop must be between 0 and 1; got height={height}"
            )
        super().__init__()
        self.height: float = height

    def forward(
        self, img_batch: torch.Tensor, label_batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        top = torch.rand(1).item() * (1 - self.height)
        _, _, H, W = img_batch.shape

        top_px, left_px, height_px, width_px = (
            round(H * top),
            0,
            round(H * self.height),
            W,
        )
        img_batch_cropped = F.crop(
            img_batch, top=top_px, left=left_px, height=height_px, width=width_px
        )
        label_batch_cropped = self._filter_label_batch(label_batch, top)
        return img_batch_cropped, label_batch_cropped

    def _filter_label_batch(
        self, label_batch: List[torch.Tensor], top: float
    ) -> List[torch.Tensor]:
        """
        If a cell is on the cropping border, how do we choose where to move the label to?

        [class xc yc w h]

        If xc in "crop region", what should we do?
        - adjust xc, yc, w, h so
        """
        filtered_label_batch = []
        for labels in label_batch:
            # yc \in [top, top + height]
            if labels.nelement() == 0:
                filtered_label_batch.append(labels)
                continue

            mask = torch.logical_and(
                top < labels[:, 2], labels[:, 2] < (top + self.height)
            )
            indices = torch.nonzero(mask)
            indices = torch.squeeze(indices, dim=1)

            filtered_labels = labels[indices, :]

            # renormalize yc, h
            filtered_labels[:, 2] = (filtered_labels[:, 2] - top) / self.height
            filtered_labels[:, 4] *= 1 / self.height

            filtered_label_batch.append(filtered_labels)

        return filtered_label_batch

    def _trim_labels(self, labels, top):
        xyxy_filtered = torchvision.ops.box_convert(labels[:, 1:], "cxcywh", "xyxy")

        xyxy_filtered[:, 1] = torch.maximum(
            xyxy_filtered[:, 1], top * torch.ones_like(xyxy_filtered[:, 1]),
        )

        xyxy_filtered[:, 3] = torch.minimum(
            xyxy_filtered[:, 3],
            (top + self.height) * torch.ones_like(xyxy_filtered[:, 3]),
        )

        cxcywh_filtered = torchvision.ops.box_convert(xyxy_filtered, "xyxy", "cxcywh",)

        labels[:, 1:] = cxcywh_filtered
        return labels


class RandomHorizontalFlipWithBBs(DualInputModule):
    """Random HFLIP that will flip the labels if the image is flipped!"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(
        self, img_batch: torch.Tensor, label_batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Expecting labels w/ form (class, xc, yc, w, h) w/ normalized coords
        """
        if torch.rand(1) < self.p:
            for labels in label_batch:
                if len(labels) > 0:
                    labels[:, 1] = 1 - labels[:, 1]
            return F.hflip(img_batch), label_batch
        return img_batch, label_batch


class RandomVerticalFlipWithBBs(DualInputModule):
    """Random VFLIP that will flip the labels if the image is flipped!"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(
        self, img_batch: torch.Tensor, label_batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Expecting labels w/ form (class, xc, yc, w, h) w/ normalized coords
        """
        if torch.rand(1) < self.p:
            for labels in label_batch:
                if len(labels) > 0:
                    labels[:, 2] = 1 - labels[:, 2]
            return F.vflip(img_batch), label_batch
        return img_batch, label_batch
