import torch
import torchvision.transforms.functional as F

from typing import Sequence, Tuple, List, Any, cast


class DualInputModule(torch.nn.Module):
    def forward(self, inpt_a, inpt_b):
        ...


class MultiArgSequential(torch.nn.Sequential):
    def __init__(self, *args: DualInputModule, **kwargs):
        super().__init__(*args, **kwargs)

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
    """ Random crop of a horizontal strip over the batch. """

    def __init__(self, height: float):
        if not (0 < height < 1):
            raise ValueError(
                f"height for RandomCrop must be between 0 and 1; got height={height}"
            )
        self.height = height

    def forward(
        self, img_batch: torch.Tensor, label_batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        top = torch.rand(1) * (1 - height)
        N, C, H, W = img_batch.shape
        img_batch_cropped = F.crop(
            img_batch, top=H * top, left=0, height=H * self.height, width=W
        )
        label_batch_cropped = self._filter_label_batch(label_batch, top, H)
        return img_batch_cropped, label_batch_cropped

    def _filter_label_batch(
        self, label_batch: List[torch.Tensor], top: float, H: int
    ) -> List[torch.Tensor]:
        """
        If a cell is on the cropping border, how do we choose where to move the label to?

        If xc in "crop region", what should we do?
        - adjust xc, yc, w, h so
        """
        filtered_label_batch = []
        for labels in label_batch:
            mask = torch.logical_not(H * top < labels[:, 2] < H * (top + self.height))
            indices = torch.nonzero(mask)
            filtered_labels = labels[indices]

            xyxy_labels = torchvision.ops.box_convert(
                filtered_labels[:, 1:], "cxcywh", "xyxy"
            )

            xyxy_filtered = torch.maximum(
                xyxy_labels[:, 1],
                H * top,
            )

            xyxy_filtered = torch.minimum(
                xyxy_filtered[:, 3],
                H * (top + self.height),
            )

            cxcywh_filtered = torchvision.ops.box_convert(
                xyxy_filtered,
                "xyxy",
                "cxcywh",
            )

            filtered_label_batch.append(cxcywh_filtered)

        return filtered_label_batch


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
