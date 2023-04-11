#! /usr/bin/env python3

import os
import wandb
import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

from lion_pytorch import Lion

from pathlib import Path
from copy import deepcopy
from typing_extensions import TypeAlias
from typing import Optional, Tuple, cast, Literal, Iterator, List

from yogo.model import YOGO
from yogo.model_funcs import get_model_func
from yogo.yogo_loss import YOGOLoss
from yogo.argparsers import train_parser
from yogo.utils import draw_rects, get_wandb_confusion, Metrics
from yogo.dataloader import (
    YOGO_CLASS_ORDERING,
    load_dataset_description,
    get_dataloader,
)
from yogo.cluster_anchors import best_anchor


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


def checkpoint_model(
    model, epoch, optimizer, name, step, model_version: Optional[str] = None, **kwargs
):
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": deepcopy(model.state_dict()),
            "optimizer_state_dict": deepcopy(optimizer.state_dict()),
            "model_version": model_version,
            **kwargs,
        },
        str(name),
    )


def get_optimizer(
    optimizer_type: Literal["lion", "adam"],
    parameters: Iterator[torch.Tensor],
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optimizer_type == "lion":
        return Lion(
            parameters, lr=learning_rate, weight_decay=weight_decay, betas=(0.95, 0.98)
        )
    elif optimizer_type == "adam":
        return AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"got invalid optimizer_type {optimizer_type}")


def train():
    config = wandb.config
    device = config["device"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    anchor_w = config["anchor_w"]
    anchor_h = config["anchor_h"]
    class_names = config["class_names"]
    weight_decay = config["weight_decay"]
    classify = not config["no_classify"]
    model = get_model_func(config["model"])
    num_classes = len(class_names)

    val_metrics = Metrics(
        num_classes=num_classes,
        device=device,
        class_names=class_names,
        classify=classify,
    )
    test_metrics = Metrics(
        num_classes=num_classes,
        device=device,
        class_names=class_names,
        classify=classify,
    )

    net = None
    if config.pretrained_path:
        print(f"loading pretrained path from {config.pretrained_path}")
        net, global_step = YOGO.from_pth(config.pretrained_path)
        net.to(device)
        if any(net.img_size.cpu().numpy() != config["resize_shape"]):
            raise RuntimeError(
                "mismatch in pretrained network image resize shape and current resize shape: "
                f"pretrained network resize_shape = {net.img_size}, requested resize_shape = {config['resize_shape']}"
            )
    else:
        net = YOGO(
            num_classes=num_classes,
            img_size=config["resize_shape"],
            anchor_w=anchor_w,
            anchor_h=anchor_h,
            model_func=model,
        ).to(device)
        global_step = 0

    print("created network")

    # TODO: generalize so we can tune Sx / Sy!
    # TODO: best way to make model architecture tunable?
    Sx, Sy = net.get_grid_size()
    wandb.config.update({"Sx": Sx, "Sy": Sy})

    print("initializing dataset...")
    (
        model_save_dir,
        train_dataloader,
        validate_dataloader,
        test_dataloader,
    ) = init_dataset(config, Sx, Sy)
    print("dataset initialized...")

    Y_loss = YOGOLoss(classify=classify).to(device)
    optimizer = get_optimizer(
        config["optimizer_type"],
        parameters=net.parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    print("created loss and optimizer")

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_dataloader),
        eta_min=learning_rate / config["decay_factor"],
    )

    print("starting training")

    best_mAP = 0
    for epoch in range(epochs):
        # train
        for imgs, labels in train_dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = net(imgs)

            loss = Y_loss(outputs, labels)

            loss.backward()

            optimizer.step()
            scheduler.step()

            global_step += 1
            wandb.log(
                {
                    "train loss": loss.item(),
                    "epoch": epoch,
                    "LR": scheduler.get_last_lr()[0],
                },
                commit=False,
                step=global_step,
            )

            wandb.log({"training grad norm": net.grad_norm()}, step=global_step)
            wandb.log({"training param norm": net.param_norm()}, step=global_step)

        # do validation things
        val_loss = 0.0
        net.eval()
        with torch.no_grad():
            for imgs, labels in validate_dataloader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = net(imgs)
                loss = Y_loss(outputs, labels)
                val_loss += loss.item()

            # just use the final imgs and labels for val!
            annotated_img = wandb.Image(
                draw_rects(
                    (
                        (255 * imgs[0, 0, ...].detach()).int()
                        if config["normalize_images"]
                        else imgs[0, 0, ...].detach()
                    ),
                    outputs[0, ...].detach(),
                    thresh=0.5,
                    labels=class_names,
                )
            )

            mAP, confusion_data, precision, recall = val_metrics.forward(
                outputs.detach(), labels.detach()
            )

            wandb.log(
                {
                    "validation bbs": annotated_img,
                    "val loss": val_loss / len(validate_dataloader),
                    "val mAP": mAP["map"],
                    "val confusion": get_wandb_confusion(
                        confusion_data, class_names, "validation confusion matrix"
                    ),
                    "val precision": precision,
                    "val recall": recall,
                },
                step=global_step,
            )

            # TODO we should choose better conditions here - e.g. mAP for no-classify isn't great,
            # and maybe we care about recall more than mAP
            if mAP["map"] > best_mAP:
                best_mAP = mAP["map"]
                wandb.log({"best_mAP_save": mAP["map"]}, step=global_step)
                checkpoint_model(
                    net,
                    epoch,
                    optimizer,
                    model_save_dir / "best.pth",
                    global_step,
                    model_version=config["model"],
                )
            else:
                checkpoint_model(
                    net,
                    epoch,
                    optimizer,
                    model_save_dir / "latest.pth",
                    global_step,
                    model_version=config["model"],
                )

        net.train()

    # do test things
    # When testing, we load the `best.pth` model.
    # That model is what we are going to use, so it is what
    # we care about!!
    net, global_step = YOGO.from_pth(model_save_dir / "best.pth")
    net.to(device)
    net.eval()

    test_loss = 0.0
    with torch.no_grad():
        for imgs, labels in test_dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = net(imgs)
            loss = Y_loss(outputs, labels)
            test_loss += loss.item()
            test_metrics.update(outputs.detach(), labels.detach())

        mAP, confusion_data, precision, recall = test_metrics.compute()
        test_metrics.reset()

        wandb.summary["test loss"] = test_loss / len(test_dataloader)
        wandb.summary["test mAP"] = mAP["map"]
        wandb.summary["test precision"] = precision
        wandb.summary["test recall"] = recall
        wandb.log(
            {
                "test confusion": get_wandb_confusion(
                    confusion_data, class_names, "test confusion matrix"
                )
            }
        )

WandbConfig: TypeAlias = dict


def init_dataset(config: WandbConfig, Sx, Sy):
    dataloaders = get_dataloader(
        config["dataset_descriptor_file"],
        config["batch_size"],
        Sx=Sx,
        Sy=Sy,
        device=config["device"],
        preprocess_type=config["preprocess_type"],
        vertical_crop_size=config["vertical_crop_size"],
        resize_shape=config["resize_shape"],
        normalize_images=config["normalize_images"],
    )

    train_dataloader = dataloaders["train"]
    validate_dataloader = dataloaders["val"]
    test_dataloader = dataloaders["test"]

    wandb.config.update(
        {  # we do this here b.c. batch_size can change wrt sweeps
            "training set size": f"{len(train_dataloader.dataset)} images",
            "validation set size": f"{len(validate_dataloader.dataset)} images",
            "testing set size": f"{len(test_dataloader.dataset)} images",
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


def do_training(args) -> None:
    """ responsible for parsing args and starting a training run
    """
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    epochs = args.epochs or 64
    batch_size = args.batch_size or 32
    learning_rate = args.lr or 3e-4
    weight_decay = args.weight_decay or 1e-2
    optimizer_type = args.optimizer or "adam"

    preprocess_type: Optional[str]
    vertical_crop_size: Optional[float] = None

    if args.crop:
        vertical_crop_size = cast(float, args.crop)
        if not (0 < vertical_crop_size < 1):
            raise ValueError(
                "vertical_crop_size must be between 0 and 1; got {vertical_crop_size}"
            )
        resize_target_size = (round(vertical_crop_size * 772), 1032)
        preprocess_type = "crop"
    elif args.resize:
        resize_target_size = cast(Tuple[int, int], tuple(args.resize))
        preprocess_type = "resize"
    else:
        resize_target_size = (772, 1032)
        preprocess_type = None

    print("loading dataset description")
    class_names, dataset_paths, _ = load_dataset_description(
        args.dataset_descriptor_file
    )

    print("getting best anchor")
    anchor_w, anchor_h = best_anchor([d["label_path"] for d in dataset_paths])

    print("initting wandb")
    wandb.init(
        project="yogo",
        entity="bioengineering",
        config={
            "optimizer_type": optimizer_type,
            "learning_rate": learning_rate,
            "decay_factor": args.lr_decay_factor,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": str(device),
            "anchor_w": anchor_w,
            "anchor_h": anchor_h,
            "model": args.model,
            "resize_shape": resize_target_size,
            "vertical_crop_size": vertical_crop_size,
            "preprocess_type": preprocess_type,
            "class_names": YOGO_CLASS_ORDERING,
            "pretrained_path": args.from_pretrained,
            "no_classify": args.no_classify,
            "run group": args.group,
            "normalize_images": args.normalize_images,
            "dataset_descriptor_file": args.dataset_descriptor_file,
            "slurm-job-id": os.getenv("SLURM_JOB_ID", default=None),
        },
        name=args.name,
        notes=args.note,
        tags=["v0.0.3"],
    )

    train()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = train_parser()
    args = parser.parse_args()

    do_training(args)
