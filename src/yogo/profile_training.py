#! /usr/bin/env python3


import torch

import torch
from torch import nn
from torch.optim import AdamW
from torch.multiprocessing import set_start_method
from torch.profiler import profile, ProfilerActivity, record_function

from yogo.model import YOGO
from yogo.argparsers import train_parser
from yogo.yogo_loss import YOGOLoss
from yogo.utils import draw_rects, Metrics
from yogo.dataloader import load_dataset_description, get_dataloader
from yogo.cluster_anchors import best_anchor, get_dataset_bounding_boxes

from pathlib import Path
from copy import deepcopy
from typing import List

WARMUP = 10
ADAM_LR = 3e-4
BATCH_SIZE = 32


class MockedModel(YOGO):
    def forward(self, *args, **kwargs):
        with record_function("MODEL FORWARD"):
            return super().forward(*args, **kwargs)


class MockedLoss(YOGOLoss):
    def forward(self, *args, **kwargs):
        with record_function("LOSS FORWARD"):
            return super().forward(*args, **kwargs)

    @classmethod
    def format_labels(cls, *args, **kwargs):
        with record_function("LOSS FORMAT LABELS"):
            return super().format_labels(*args, **kwargs)


def profile_run(
    dev,
    train_dataloader,
    validate_dataloader,
    test_dataloader,
    anchor_w,
    anchor_h,
    img_size,
    class_names,
):
    num_classes = len(class_names)
    net = MockedModel(
        img_size=img_size, anchor_w=anchor_w, anchor_h=anchor_h, num_classes=num_classes
    ).to(dev)
    Y_loss = MockedLoss().to(dev)
    optimizer = AdamW(net.parameters(), lr=ADAM_LR)
    metrics = Metrics(num_classes=num_classes, device=dev, class_names=class_names)

    Sx, Sy = net.get_grid_size(img_size)

    print("warming up")
    for epoch in range(WARMUP):
        outputs = net(torch.randn(1, 1, *img_size, device=dev))
    net.zero_grad()

    print("here we goooooo!")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for imgs, labels in train_dataloader:
            optimizer.zero_grad(set_to_none=True)

            outputs = net(imgs)
            formatted_labels = MockedLoss.format_labels(
                outputs, labels, num_classes=num_classes, device=device
            )
            loss = Y_loss(outputs, formatted_labels)
            loss.backward()
            optimizer.step()

            metrics.update(outputs, formatted_labels)

        metrics.compute()
        return prof


if __name__ == "__main__":
    set_start_method("spawn")

    parser = train_parser()
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    class_names, dataset_paths, _ = load_dataset_description(
        args.dataset_descriptor_file
    )
    label_paths = [d["label_path"] for d in dataset_paths]
    anchor_w, anchor_h = best_anchor(
        get_dataset_bounding_boxes(label_paths, center_box=True)
    )

    # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner
    torch.backends.cudnn.benchmark = True
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True

    vertical_crop_size = 0.25
    resize_target_size = (round(vertical_crop_size * 772), 1032)

    dataloaders = get_dataloader(
        args.dataset_descriptor_file,
        BATCH_SIZE,
        device=device,
        vertical_crop_size=vertical_crop_size,
    )

    train_dataloader = dataloaders["train"]
    validate_dataloader = dataloaders["val"]
    test_dataloader = dataloaders["test"]

    prof = profile_run(
        device,
        train_dataloader,
        validate_dataloader,
        test_dataloader,
        anchor_w,
        anchor_h,
        resize_target_size,
        class_names,
    )

    prof.export_chrome_trace("chrome_profile.json")
    print("I am done!")
