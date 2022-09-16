#! /usr/bin/env python3


import torch

import torch
from torch import nn
from torch.optim import AdamW
from torch.multiprocessing import set_start_method
from torch.profiler import profile, ProfilerActivity, record_function

from model import YOGO
from argparser import parse
from yogo_loss import YOGOLoss
from utils import draw_rects, Metrics
from dataloader import load_dataset_description, get_dataloader
from cluster_anchors import best_anchor, get_dataset_bounding_boxes

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
    net = MockedModel(img_size=img_size, anchor_w=anchor_w, anchor_h=anchor_h).to(dev)
    Y_loss = MockedLoss().to(dev)
    optimizer = AdamW(net.parameters(), lr=ADAM_LR)
    metrics = Metrics(num_classes=4, device=dev, class_names=class_names)

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
                outputs,
                labels,
                device=device
            )
            loss = Y_loss(outputs, formatted_labels)
            loss.backward()
            optimizer.step()

            metrics.update(outputs, formatted_labels)

        metrics.compute()
        return prof


if __name__ == "__main__":
    set_start_method("spawn")

    args = parse()

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
        get_dataset_bounding_boxes(label_paths, center_box=True), kmeans=True
    )

    # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner
    torch.backends.cudnn.benchmark = True
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True

    # TODO: EPOCH and BATCH_SIZE and img_size in yml file?
    resize_target_size = (300, 400)
    dataloaders = get_dataloader(
        args.dataset_descriptor_file,
        BATCH_SIZE,
        img_size=resize_target_size,
        device=device,
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
