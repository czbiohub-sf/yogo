#! /usr/bin/env python3

import wandb
import torch

import torch
from torch import nn
from torch.optim import AdamW
from torch.multiprocessing import set_start_method
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingWarmRestarts

from train import init_dataset, dataloader_class_weights
from model import YOGO
from argparser import parse
from yogo_loss import YOGOLoss
from utils import draw_rects, Metrics
from dataloader import load_dataset_description, get_dataloader
from cluster_anchors import best_anchor, get_dataset_bounding_boxes

from pathlib import Path
from copy import deepcopy
from typing import List


# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner
torch.backends.cudnn.benchmark = True
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True


def pareto_quality():
    report_periods = (16, 32, 64, 128, 256)

    config = wandb.config
    device = config["device"]
    anchor_w = config["anchor_w"]
    anchor_h = config["anchor_h"]
    class_names = config["class_names"]

    (
        model_save_dir,
        train_dataloader,
        validate_dataloader,
        test_dataloader,
    ) = init_dataset(config)

    weights = dataloader_class_weights(train_dataloader)

    net = YOGO(
        img_size=config["resize_shape"],
        anchor_w=anchor_w,
        anchor_h=anchor_h,
        device=device,
    )
    Y_loss = YOGOLoss(class_weights=weights, device=device)
    optimizer = AdamW(net.parameters(), lr=config["learning_rate"])

    min_period = 8 * len(train_dataloader)
    lin = LinearLR(optimizer, start_factor=0.01, end_factor=1, total_iters=min_period)
    cs = CosineAnnealingWarmRestarts(optimizer, T_0=min_period, T_mult=2)
    scheduler = SequentialLR(optimizer, [lin, cs], [min_period])

    metrics = Metrics(num_classes=4, device=device, class_names=class_names)

    # TODO: generalize so we can tune Sx / Sy!
    # TODO: best way to make model architecture tunable?
    Sx, Sy = net.get_grid_size(config["resize_shape"])
    wandb.config.update({"Sx": Sx, "Sy": Sy})

    global_step = 0
    for epoch in range(config["epochs"]):
        # train
        for imgs, labels in train_dataloader:
            global_step += 1

            optimizer.zero_grad(set_to_none=True)

            outputs = net(imgs)
            formatted_labels = Y_loss.format_labels(outputs, labels, device=device)
            loss = Y_loss(outputs, formatted_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if global_step % 100 == 0:
                wandb.log(
                    {
                        "train loss": loss.item(),
                        "LR": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    },
                    commit=False,
                    step=global_step,
                )

        net.eval()

        if (epoch + 1) in report_periods:
            # do validation things
            val_loss = 0.0
            for imgs, labels in validate_dataloader:
                with torch.no_grad():
                    outputs = net(imgs)
                    formatted_labels = Y_loss.format_labels(
                        outputs, labels, device=device
                    )
                    loss = Y_loss(outputs, formatted_labels)
                    val_loss += loss.item()

                metrics.update(outputs, formatted_labels)

            mAP, _ = metrics.compute()
            metrics.reset()

            wandb.log(
                {
                    "val loss": val_loss / len(validate_dataloader),
                    "val mAP": mAP["map"],
                    "epoch": epoch,
                },
                step=global_step,
            )
        net.train()


if __name__ == "__main__":
    args = parse()

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    epochs = 256
    adam_lr = 3e-4
    batch_size = 32
    resize_target_size = (300, 400)

    class_names, dataset_paths, _ = load_dataset_description(
        args.dataset_descriptor_file
    )
    label_paths = [d["label_path"] for d in dataset_paths]
    anchor_w, anchor_h = best_anchor(
        get_dataset_bounding_boxes(label_paths, center_box=True), kmeans=True
    )

    wandb.init(
        "yogo",
        entity="bioengineering",
        config={
            "learning_rate": adam_lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": str(device),
            "anchor_w": anchor_w,
            "anchor_h": anchor_h,
            "resize_shape": resize_target_size,
            "class_names": class_names,
            "run group": args.group,
            "dataset_descriptor_file": args.dataset_descriptor_file,
        },
        notes="pareto run: " + args.note,
        tags=["v0.0.1"],
    )

    pareto_quality()
