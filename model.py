import torch
from torch import nn


class PrintLayer(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


class YOGO(nn.Module):
    def __init__(self, num_anchors: int):
        super().__init__()
        self.num_anchors = num_anchors
        self.backbone = self.gen_backbone()
        self.head = self.gen_head(
            num_channels=1024, num_classes=4, num_anchors=num_anchors, Sx=13, Sy=13
        )

    def num_params(self):
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

    def gen_head(
        self, num_channels: int, num_classes: int, num_anchors: int, Sx: int, Sy: int
    ) -> nn.Module:
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
        conv_block_3 = nn.Conv2d(num_channels, (5 + num_classes) * num_anchors, 1)
        return nn.Sequential(conv_block_1, conv_block_2, conv_block_3)

    def forward(self, x: torch.Tensor):
        x = x.float()
        x = self.backbone(x)
        x = self.head(x)
        bs, preds, sx, sy = x.shape
        # TODO: this is a sanity check, should be redundant
        assert preds / self.num_anchors == preds // self.num_anchors
        return x.view(bs, self.num_anchors, preds // self.num_anchors, sx, sy)


if __name__ == "__main__":
    Y = YOGO(1)
    x = torch.randn(5, 1, 150, 200)
    import time

    t0 = time.perf_counter()
    N = 10
    for _ in range(N):
        Y(x)
    print((time.perf_counter() - t0) / N)
    print(Y(x).shape, torch.prod(torch.tensor(Y(x).shape)))
