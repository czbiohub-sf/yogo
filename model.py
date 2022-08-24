import torch
from torch import nn


class YOGO(nn.Module):
    """
    Restricting assumptions:
        - all objects being detected are roughly the same size (K-Means Clustering anchor
        boxes across the dataset verifies this), meaning that it does not make sense to
        have more than 1 anchor box
        - grayscale

    TODO:
        - Figure out conv layer sizing to properly reduce size of input to desired Sx, Sy
        - Add residuals?
    """
    def __init__(self, anchor_w, anchor_h):
        super().__init__()
        self.num_anchors = 1
        self.anchor_w = anchor_w
        self.anchor_h = anchor_h
        self.backbone = self.gen_backbone()
        self.head = self.gen_head(num_channels=1024, num_classes=4)
        self.device = 'cpu'

    def to(self, device):
        # FIXME: hack?
        self.device = device
        super().to(device)
        return self

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def gen_backbone(self) -> nn.Module:
        conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        conv_block_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        conv_block_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        conv_block_4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        conv_block_5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        conv_block_6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        conv_block_7 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
        )
        return nn.Sequential(
            conv_block_1,
            conv_block_2,
            conv_block_3,
            conv_block_4,
            conv_block_5,
            conv_block_6,
            conv_block_7,
        )

    def gen_head(self, num_channels: int, num_classes: int) -> nn.Module:
        conv_block_1 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(),
        )
        conv_block_2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(),
        )
        conv_block_3 = nn.Conv2d(num_channels, (5 + num_classes), 1)
        return nn.Sequential(conv_block_1, conv_block_2, conv_block_3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: do output transformations for inference
        # TODO: better to use "output transformation" inference or raw preds, and
        # backprop on raw network outputs?
        x = x.float()
        x = self.backbone(x)
        x = self.head(x)

        bs, preds, sy, sx = x.shape

        Cxs = torch.linspace(0, 1 - 1/sx, sx).expand(sy, -1).to(self.device)
        Cys = torch.linspace(0, 1 - 1/sy, sy).expand(1,-1).T.expand(sy,sx).to(self.device)

        # implementation of "Direct Location Prediction" from YOLO9000 paper
        # Order of meanings:
        #  center of bounding box in x
        #  center of bounding box in y
        #  width of bounding box
        #  height of bounding box
        #  'objectness' score
        x[:, 0, :, :] = torch.sigmoid(x[:, 0, :, :]) + Cxs
        x[:, 1, :, :] = torch.sigmoid(x[:, 1, :, :]) + Cys
        x[:, 2, :, :] = self.anchor_w * torch.exp(x[:, 2, :, :])
        x[:, 3, :, :] = self.anchor_h * torch.exp(x[:, 3, :, :])
        x[:, 4, :, :] = torch.sigmoid(x[:, 4, :, :])
        if not self.training:
            # TODO: If using cross entropy loss, I *think* that we
            # avoid applying any softmax / logsoftmax
            x[:, 5:, :, :] = torch.softmax(x[:, 5:, :, :], dim=1)

        return x


if __name__ == "__main__":
    Y = YOGO(0.0455, 0.059)
    Y.eval()
    x = torch.randn(1, 1, 416, 416)
    import time

    t0 = time.perf_counter()
    N = 10
    for _ in range(N):
        Y(x)
    print((time.perf_counter() - t0) / N)
    print(Y(x).shape, Y(x)[0, :, 0, 0])
