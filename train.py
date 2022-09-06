#! /usr/bin/env python3


import wandb
import torch

import torch
from torch import nn
from torch.optim import AdamW

from model import YOGO
from argparser import parse
from yogo_loss import YOGOLoss
from utils import draw_rects, batch_mAP
from dataloader import load_dataset_description, get_dataloader
from cluster_anchors import best_anchor, get_dataset_bounding_boxes

from pathlib import Path
from copy import deepcopy
from typing import List


EPOCHS = 64
ADAM_LR = 3e-4
BATCH_SIZE = 16

# TODO find sync points - wandb may be it, unfortunately :(
# https://pytorch.org/docs/stable/generated/torch.cuda.set_sync_debug_mode.html#torch-cuda-set-sync-debug-mode
# this will error out if a synchronizing operation occurs
#
# TUNING GUIDE - goes over this
# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

# TODO
# measure forward / backward pass timing w/
# https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution

# TODO test! seems like potentially large improvement on the table
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True


def train(
    dev,
    train_dataloader,
    validate_dataloader,
    test_dataloader,
    anchor_w,
    anchor_h,
    img_size,
):
    net = YOGO(img_size=img_size, anchor_w=anchor_w, anchor_h=anchor_h).to(dev)
    Y_loss = YOGOLoss().to(dev)
    optimizer = AdamW(net.parameters(), lr=ADAM_LR)

    Sx, Sy = net.get_grid_size(img_size)
    wandb.config.update(
        {
            "Sx": Sx,
            "Sy": Sy,
        }
    )

    if wandb.run.name is not None:
        model_save_dir = Path(f"trained_models/{wandb.run.name}")
    else:
        model_save_dir = Path(
            f"trained_models/unamed_run_{torch.randint(100, size=(1,)).item()}"
        )
    model_save_dir.mkdir(exist_ok=True, parents=True)

    global_step = 0
    for epoch in range(EPOCHS):
        for i, (imgs, labels) in enumerate(train_dataloader, 1):
            global_step += 1

            optimizer.zero_grad(set_to_none=True)

            outputs = net(imgs)
            loss = Y_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log(
                {"train loss": loss.item(), "epoch": epoch},
                commit=False,
                step=global_step,
            )

        val_loss = 0.0
        net.eval()
        for data in validate_dataloader:
            imgs, labels = data

            with torch.no_grad():
                outputs = net(imgs)
                loss = Y_loss(outputs, labels)
                val_loss += loss.item()

        # just use final batch from validate_dataloader for now!
        annotated_img = wandb.Image(
            draw_rects(imgs[0, 0, ...], outputs[0, ...], thresh=0.5)
        )
        mAP_calcs = batch_mAP(
            outputs, YOGOLoss.format_label_batch(outputs, labels, device=device)
        )
        wandb.log(
            {
                "validation bbs": annotated_img,
                "val loss": val_loss / len(validate_dataloader),
                "val mAP": mAP_calcs["map"],
            },
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": deepcopy(net.state_dict()),
                "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                "avg_val_loss": val_loss / len(validate_dataloader),
            },
            str(model_save_dir / f"{wandb.run.name}_{epoch}_{i}.pth"),
        )
        net.train()

    net.eval()
    test_loss = 0.0
    for data in test_dataloader:
        imgs, labels = data

        with torch.no_grad():
            outputs = net(imgs)
            loss = Y_loss(outputs, labels)
            test_loss += loss.item()

    wandb.log(
        {
            "test loss": test_loss / len(test_dataloader),
        },
    )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": deepcopy(net.state_dict()),
            "optimizer_state_dict": deepcopy(optimizer.state_dict()),
            "avg_val_loss": val_loss / len(validate_dataloader),
        },
        str(model_save_dir / f"{wandb.run.name}_{epoch}_{i}.pth"),
    )


if __name__ == "__main__":
    args = parse()

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    _, dataset_paths, __ = load_dataset_description(args.dataset_descriptor_file)
    # just pick a random dataset for now - ideally, we go over the entire set of datasets
    label_paths = [d["label_path"] for d in dataset_paths]
    anchor_w, anchor_h = best_anchor(
        get_dataset_bounding_boxes(label_paths, center_box=True), kmeans=True
    )

    # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner
    torch.backends.cudnn.benchmark = True

    # TODO: BATCH_SIZE and img_size in yml file?
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

    wandb.init(
        "yogo",
        entity="ajacobsen-czb",
        config={
            "learning rate": ADAM_LR,
            "epochs": EPOCHS,
            "batch size": BATCH_SIZE,
            "training set size": len(train_dataloader),
            "device": str(device),
            "anchor w": anchor_w,
            "anchor h": anchor_h,
            "resize shape": resize_target_size,
            "run group": args.group,
        },
        notes=args.note,
        tags=["initial-testing"],
    )

    train(
        device,
        train_dataloader,
        validate_dataloader,
        test_dataloader,
        anchor_w,
        anchor_h,
        resize_target_size,
    )
