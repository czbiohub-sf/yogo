#! /usr/bin/env python3


import wandb
import torch

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from model import YOGO
from argparsers import train_parser
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


def train():
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
        img_size=config["resize_shape"],
        anchor_w=anchor_w,
        anchor_h=anchor_h,
    ).to(device)
    Y_loss = YOGOLoss().to(device)
    optimizer = AdamW(net.parameters(), lr=config["learning_rate"])
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.3,
        total_steps=len(train_dataloader) * config["epochs"],
    )
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
            formatted_labels = YOGOLoss.format_labels(outputs, labels, device=device)
            loss = Y_loss(outputs, formatted_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            wandb.log(
                {"train loss": loss.item(), "epoch": epoch, "LR": scheduler.get_last_lr()},
                commit=False,
                step=global_step,
            )

        wandb.log({"training grad norm": net.grad_norm()}, step=global_step)

        # do validation things
        val_loss = 0.0
        net.eval()
        for imgs, labels in validate_dataloader:
            with torch.no_grad():
                outputs = net(imgs)
                formatted_labels = YOGOLoss.format_labels(
                    outputs, labels, device=device
                )
                loss = Y_loss(outputs, formatted_labels)
                val_loss += loss.item()

            metrics.update(outputs, formatted_labels)

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
                "val confusion": get_wandb_confusion(
                    confusion_data, "validation confusion matrix"
                ),
            },
        )

        if mAP["map"] > best_mAP:
            best_mAP = mAP["map"]
            wandb.log({"best_mAP_save": mAP["map"]}, step=global_step)
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": deepcopy(net.state_dict()),
                    "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                },
                str(model_save_dir / f"best.pth"),
            )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": deepcopy(net.state_dict()),
                "optimizer_state_dict": deepcopy(optimizer.state_dict()),
            },
            str(model_save_dir / f"latest.pth"),
        )

        net.train()

    # do test things
    net.eval()
    test_loss = 0.0
    for imgs, labels in test_dataloader:
        with torch.no_grad():
            outputs = net(imgs)
            formatted_labels = YOGOLoss.format_labels(outputs, labels, device=device)
            loss = Y_loss(outputs, formatted_labels)
            test_loss += loss.item()

        metrics.update(outputs, formatted_labels)

    mAP, confusion_data = metrics.compute()
    metrics.reset()
    wandb.log(
        {
            "test loss": test_loss / len(test_dataloader),
            "test mAP": mAP["map"],
            "test confusion": get_wandb_confusion(
                confusion_data, "test confusion matrix"
            ),
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


def init_dataset(config):
    dataloaders = get_dataloader(
        config["dataset_descriptor_file"],
        config["batch_size"],
        img_size=config["resize_shape"],
        device=config["device"],
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


def get_wandb_confusion(confusion_data, title):
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
        {"title": title},
    )


if __name__ == "__main__":
    parser = train_parser()
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    epochs = 192
    adam_lr = 3e-4
    batch_size = 32
    resize_target_size = (600, 800)

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
        notes=args.note,
        tags=["v0.0.1"],
    )

    train()
