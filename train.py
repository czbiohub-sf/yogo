#! /usr/bin/env python3


import torch
from torch import nn
from torch.optim import AdamW

import wandb
from model import YOGO
from utils import draw_rects
from yogo_loss import YOGOLoss
from dataloader import get_dataloader

from pathlib import Path
from copy import deepcopy
from typing import List


EPOCHS = 256
ADAM_LR = 3e-4
BATCH_SIZE = 16
VALIDATION_PERIOD = 100
anchor_h = 17 / 300
anchor_w = 17 / 400

# TODO find sync points - wandb may be it, unfortunately :(
# https://pytorch.org/docs/stable/generated/torch.cuda.set_sync_debug_mode.html#torch-cuda-set-sync-debug-mode

# TODO
# measure forward / backward pass timing w/
# https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution

# TODO test! seems like potentially large improvement on the table
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloaders = get_dataloader(
    "healthy_cell_dataset.yml",
    BATCH_SIZE,
)
train_dataloader = dataloaders["train"]
validate_dataloader = dataloaders["val"]
test_dataloader = dataloaders["test"]

wandb.init(
    "yogo",
    config={
        "learning_rate": ADAM_LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_set_size": len(train_dataloader),
        "device": str(device),
        "anchor_w": anchor_w,
        "anchor_h": anchor_h,
    },
)


def train(dev):
    net = YOGO(anchor_w=anchor_w, anchor_h=anchor_h).to(dev)
    Y_loss = YOGOLoss().to(dev)
    optimizer = AdamW(net.parameters(), lr=ADAM_LR)

    if wandb.run.name is not None:
        model_save_dir = Path(f"trained_models/{wandb.run.name}")
        model_save_dir.mkdir(exist_ok=True, parents=True)

    global_step = 0
    for epoch in range(EPOCHS):
        for i, (imgs, labels) in enumerate(train_dataloader, 1):
            global_step += 1
            imgs = imgs.to(dev)
            # labels = labels.to(dev)  # TODO: does this have meaning?!?!

            optimizer.zero_grad()  # possible set_to_none=True to get "modest" speedup

            outputs = net(imgs)
            loss = Y_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log(
                {"train_loss": loss.item(), "epoch": epoch},
                commit=False,
                step=global_step,
            )

            if global_step % VALIDATION_PERIOD == 0:
                wandb.log(
                    {
                        "training_bbs": wandb.Image(
                            draw_rects(imgs[0, 0, ...], outputs[0, ...], thresh=0.5)
                        )
                    },
                )

                val_loss = 0.0

                net.eval()
                for data in validate_dataloader:
                    imgs, labels = data
                    imgs = imgs.to(dev)

                    with torch.no_grad():
                        outputs = net(imgs)
                        loss = Y_loss(outputs, labels)
                        val_loss += loss.item()

                wandb.log(
                    {"val_loss": val_loss / len(validate_dataloader)},
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

    print("done training")
    net.eval()
    test_loss = 0.0
    for data in test_dataloader:
        imgs, labels = data
        imgs = imgs.to(dev)

        with torch.no_grad():
            outputs = net(imgs)
            loss = Y_loss(outputs, labels)
            test_loss += loss.item()

    wandb.log(
        {
            "test_loss": test_loss / len(test_dataloader),
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
    print(f"using device {device}")
    train(device)
