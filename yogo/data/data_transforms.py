import torch
import torchvision.transforms.functional as F

from typing import Tuple


""" Major TODO

This was written before TorchVision's "Transforms V2"[0] was released.
It is (probably) faster + can do all this natively. It's silly to rewrite all this!
We should refactor this out.

[0] https://pytorch.org/vision/stable/transforms.html#v2-api-reference-recommended
"""


class DualInputModule(torch.nn.Module):
    def forward(self, inpt_a, inpt_b): ...


class DualInputId(DualInputModule):
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
    good for e.g. resizes with bboxes with normalized coords
    """

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, img_batch, labels):
        return self.transform(img_batch), labels


class RandomHorizontalFlipWithBBs(DualInputModule):
    """Random HFLIP that will flip the labels if the image is flipped!"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(
        self, img_batch: torch.Tensor, label_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        labels have shape (batch size, len([obj mask *[x y x y] class]), Sy, Sx) == (batch size, 6, Sy, Sx)

        Need to flip labels around the tensor axes too!
        """
        assert img_batch.ndim == 4 and label_batch.ndim == 4
        if torch.rand(1) < self.p:
            label_batch[:, 1, :, :], label_batch[:, 3, :, :] = (
                1 - label_batch[:, 3, :, :],
                1 - label_batch[:, 1, :, :],
            )
            return F.hflip(img_batch), torch.flip(label_batch, dims=(3,))
        return img_batch, label_batch


class RandomVerticalFlipWithBBs(DualInputModule):
    """Random VFLIP that will flip the labels if the image is flipped!"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(
        self, img_batch: torch.Tensor, label_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        labels have shape (batch size, len([obj mask *[x y x y] class]), Sy, Sx) == (batch size, 6, Sy, Sx)

        Need to flip labels around the tensor axes too!
        """
        assert img_batch.ndim == 4 and label_batch.ndim == 4
        if torch.rand(1) < self.p:
            label_batch[:, 2, :, :], label_batch[:, 4, :, :] = (
                1 - label_batch[:, 4, :, :],
                1 - label_batch[:, 2, :, :],
            )
            return F.vflip(img_batch), torch.flip(label_batch, dims=(2,))
        return img_batch, label_batch
