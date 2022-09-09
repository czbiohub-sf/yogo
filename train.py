#! /usr/bin/env python3


import wandb
import torch

import torch
from torch import nn
from torch.optim import AdamW

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

# TODO find sync points
# https://pytorch.org/docs/stable/generated/torch.cuda.set_sync_debug_mode.html#torch-cuda-set-sync-debug-mode
# this will error out if a synchronizing operation occurs

# TUNING GUIDE - goes over this
# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

# TODO
# measure forward / backward pass timing w/
# https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution


def train(
    dev,
    dataset_descriptor,
    anchor_w,
    anchor_h,
    img_size,
    class_names,
):
    config = wandb.config

    train_dataloader, validate_dataloader, test_dataloader = init_dataset(
        config, dataset_descriptor
    )

    net = YOGO(
        img_size=config["resize shape"], anchor_w=anchor_w, anchor_h=anchor_h
    ).to(dev)
    Y_loss = YOGOLoss().to(dev)
    optimizer = AdamW(net.parameters(), lr=config["learning rate"])
    metrics = Metrics(num_classes=4, device=dev, class_names=class_names)

    # TODO: generalize so we can tune Sx / Sy!
    # TODO: best way to make model architecture tunable?
    Sx, Sy = net.get_grid_size(config["resize shape"])
    wandb.config.update({"Sx": Sx, "Sy": Sy})

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

            wandb.log(
                {"train loss": loss.item(), "epoch": epoch},
                commit=False,
                step=global_step,
            )

        # do validation things
        val_loss = 0.0
        net.eval()
        for imgs, labels in validate_dataloader:
            with torch.no_grad():
                outputs = net(imgs)
                loss = Y_loss(outputs, labels)
                val_loss += loss.item()

            metrics.update(
                outputs, YOGOLoss.format_labels(outputs, labels, device=device)
            )

        annotated_img = wandb.Image(
            draw_rects(imgs[0, 0, ...], outputs[0, ...], thresh=0.5)
        )

        mAP, confusion_data = metrics.compute()
        metrics.reset()

        wandb.log(
            {
                "validation bbs": annotated_img,
                "val loss": val_loss / len(validate_dataloader),
                "val mAP": mAP["map"],
                "val confusion": get_wandb_confusion(confusion_data),
            },
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": deepcopy(net.state_dict()),
                "optimizer_state_dict": deepcopy(optimizer.state_dict()),
            },
            str(model_save_dir / f"{wandb.run.name}_{epoch}_{i}.pth"),
        )
        net.train()

    # do test things
    net.eval()
    test_loss = 0.0
    for imgs, labels in test_dataloader:
        with torch.no_grad():
            outputs = net(imgs)
            loss = Y_loss(outputs, labels)
            test_loss += loss.item()

        metrics.update(outputs, YOGOLoss.format_labels(outputs, labels, device=device))

    mAP, confusion_data = metrics.compute()
    metrics.reset()
    wandb.log(
        {
            "test loss": test_loss / len(test_dataloader),
            "test mAP": mAP["map"],
            "test confusion": get_wandb_confusion(confusion_data),
        },
    )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": deepcopy(net.state_dict()),
            "optimizer_state_dict": deepcopy(optimizer.state_dict()),
        },
        str(model_save_dir / f"{wandb.run.name}_{epoch}_{i}.pth"),
    )


def init_dataset(config, dataset_descriptor_file):
    dataloaders = get_dataloader(
        dataset_descriptor_file,
        config["batch size"],
        img_size=config["resize shape"],
        device=config["device"],
    )

    train_dataloader = dataloaders["train"]
    validate_dataloader = dataloaders["val"]
    test_dataloader = dataloaders["test"]

    wandb.config.update(
        {  # we do this here b.c. batch size can change wrt sweeps
            "training set size": f"{len(train_dataloader) * config['batch size']} images",
            "validation set size": f"{len(validate_dataloader) * config['batch size']} images",
            "testing set size": f"{len(test_dataloader) * config['batch size']} images",
        }
    )

    if wandb.run.name is not None:
        model_save_dir = Path(f"trained_models/{wandb.run.name}")
    else:
        model_save_dir = Path(
            f"trained_models/unamed_run_{torch.randint(100, size=(1,)).item()}"
        )
    model_save_dir.mkdir(exist_ok=True, parents=True)

    return train_dataloader, validate_dataloader, test_dataloader


def get_wandb_confusion(confusion_data):
    return wandb.plot_table(
        "wandb/confusion_matrix/v1",
        wandb.Table(
            columns=["Actual", "Predicted", "nPredictions"],
            data=confusion_data,
        ),
        {
            "Actual": "Actual",
            "Predicted": "Predicted",
            "nPredictions": "nPredictions",
        },
        {"title": "validation confusion matrix"},
    )


if __name__ == "__main__":
    args = parse()

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    epochs = 128
    adam_lr = 3e-4
    batch_size = 16
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
        entity="ajacobsen-czb",  # aaagh! this needs to be bioengineering
        config={
            "learning rate": adam_lr,
            "epochs": epochs,
            "batch size": batch_size,
            "device": str(device),
            "anchor w": anchor_w,
            "anchor h": anchor_h,
            "resize shape": resize_target_size,
            "run group": args.group,
        },
        notes=args.note,
        tags=["v0.0.1"],
    )

    train(
        device,
        args.dataset_descriptor_file,
        anchor_w,
        anchor_h,
        resize_target_size,
        class_names,
    )
