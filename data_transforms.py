import torch
import torchvision.transforms.functional as F

from typing import List, Tuple


class RandomHorizontalFlipYOGO(torch.nn.Module):
    """Random HFLIP that will flip the labels if the image is flipped!"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(
        self, img: torch.Tensor, labels: List[List[float]]
    ) -> Tuple[torch.Tensor, List[List[float]]]:
        """
        Expecting labels w/ form (class, xc, yc, w, h) w/ normalized coords
        """
        if torch.rand(1) < self.p:
            # this math op reverses the labels.
            # flip them back to make sorting in loss function quick.
            # labels is ordered via itemgetter(1,2), so all we have to
            # do here is reverse in x.
            flipped_labels = [
                [
                    l[0],
                    1 - l[1],
                    l[2],
                    l[3],
                    l[4],
                ]
                for l in labels
            ]
            return F.hflip(img), flipped_labels
        return img, labels


class RandomVerticalFlipYOGO(torch.nn.Module):
    """Random VFLIP that will flip the labels if the image is flipped!"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(
        self, img: torch.Tensor, labels: List[List[float]]
    ) -> Tuple[torch.Tensor, List[List[float]]]:
        """
        Expecting labels w/ form (class, xc, yc, w, h) w/ normalized coords
        """
        if torch.rand(1) < self.p:
            # this math op reverses the labels.
            # flip them back to make sorting in loss function quick.
            # labels is ordered via itemgetter(1,2), so we have to sort
            # in x and then in y to maintain proper order. "x" should
            # already be ordered, so this should be relatively quick.
            flipped_labels = [
                [
                    l[0],
                    l[1],
                    1 - l[2],
                    l[3],
                    l[4],
                ]
                for l in labels
            ]
            return F.vflip(img), flipped_labels
        return img, labels
