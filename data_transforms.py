import torch
import torchvision.transforms.functional as F

from typing import List, Tuple, Any


class ImageTransformLabelIdentity(torch.nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, img_batch, labels):
        return self.transform(img_batch), labels


class RandomHorizontalFlipWithBBs(torch.nn.Module):
    """Random HFLIP that will flip the labels if the image is flipped!"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img_batch, label_batch) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Expecting labels w/ form (class, xc, yc, w, h) w/ normalized coords
        """
        if torch.rand(1) < self.p:
            for labels in label_batch:
                if len(labels) > 0:
                    labels[:, 1] = 1 - labels[:, 1]
            return F.hflip(img_batch), label_batch
        return img_batch, label_batch


class RandomVerticalFlipWithBBs(torch.nn.Module):
    """Random VFLIP that will flip the labels if the image is flipped!"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img_batch, label_batch) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Expecting labels w/ form (class, xc, yc, w, h) w/ normalized coords
        """
        if torch.rand(1) < self.p:
            for labels in label_batch:
                if len(labels) > 0:
                    labels[:, 2] = 1 - labels[:, 2]
            return F.vflip(img_batch), label_batch
        return img_batch, label_batch
