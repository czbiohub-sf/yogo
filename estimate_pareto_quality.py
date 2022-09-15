#! /usr/bin/env python3

import wandb
import torch

import torch
from torch import nn
from torch.optim import AdamW
from torch.multiprocessing import set_start_method
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingWarmRestarts

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
    report_periods = (16, 24, 40, 72, 136, 264)

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

    net = YOGO(
        img_size=config["resize_shape"], anchor_w=anchor_w, anchor_h=anchor_h
    ).to(device)
    Y_loss = YOGOLoss().to(device)
    optimizer = AdamW(net.parameters(), lr=config["learning_rate"])

    min_period = 8 * len(train_dataloader)
    lin = LinearLR(optimizer, start_factor=0.01, total_iters=min_period)
    cs = CosineAnnealingWarmRestarts(optimizer, T_0=min_period, T_mult=2)
    scheduler = SequentialLR(optimizer, [lin, cs], [min_period])

    metrics = Metrics(num_classes=4, device=device, class_names=class_names)

    # TODO: generalize so we can tune Sx / Sy!
    # TODO: best way to make model architecture tunable?
    Sx, Sy = net.get_grid_size(config["resize_shape"])
    wandb.config.update({"Sx": Sx, "Sy": Sy})

    best_mAP = 0
    global_step = 0
    for epoch in range(config["epochs"]):
        # train
        for i, (imgs, labels) in enumerate(train_dataloader, 1):
            global_step += 1

            optimizer.zero_grad(set_to_none=True)

            outputs = net(imgs)
            loss = Y_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            wandb.log(
                {
                    "train loss": loss.item()
                    "LR": : scheduler.get_last_lr()[0]
                },
                commit=False,
                step=global_step,
            )

        if (epoch + 1) in report_periods:
            net.eval()

            # do validation things
            val_loss = 0.0
            for imgs, labels in validate_dataloader:
                with torch.no_grad():
                    outputs = net(imgs)
                    loss = Y_loss(outputs, labels)
                    val_loss += loss.item()

                metrics.update(
                    outputs, YOGOLoss.format_labels(outputs, labels, device=device)
                )

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


def init_dataset(config):
    dataloaders = get_dataloader(
        config["dataset_descriptor_file"],
        config["batch_size"],
        img_size=config["resize_shape"],
        device=config["device"],
        split_fractions_override={"train": 0.8, "test": 0., "val": 0.2}
    )

    train_dataloader = dataloaders["train"]
    validate_dataloader = dataloaders["val"]
    test_dataloader = dataloaders["test"]

    wandb.config.update(
        {  # we do this here b.c. batch_size can change wrt sweeps
            "training set size": f"{len(train_dataloader) * config['batch_size']} images",
            "validation set size": f"{len(validate_dataloader) * config['batch_size']} images",
            "testing set size": f"{len(test_dataloader) * config['batch_size']} images",
        }
    )

    if wandb.run.name is not None:
        model_save_dir = Path(f"trained_models/{wandb.run.name}")
    else:
        model_save_dir = Path(
            f"trained_models/unnamed_run_{torch.randint(100, size=(1,)).item()}"
        )
    model_save_dir.mkdir(exist_ok=True, parents=True)

    return model_save_dir, train_dataloader, validate_dataloader, test_dataloader


if __name__ == "__main__":
    args = parse()

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    epochs = 264
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
