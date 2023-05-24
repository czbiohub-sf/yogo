#! /usr/bin/env python3

import os
import wandb
import torch
import queue
import warnings

import torch.multiprocessing as mp

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from lion_pytorch import Lion

from pathlib import Path
from copy import deepcopy
from itertools import cycle
from typing_extensions import TypeAlias
from typing import Optional, Tuple, cast, Literal, Iterator

from yogo.utils.default_hyperparams import DefaultHyperparams as df

from yogo.model import YOGO
from yogo.model_defns import get_model_func
from yogo.yogo_loss import YOGOLoss
from yogo.metrics import Metrics
from yogo.data.dataset import YOGO_CLASS_ORDERING
from yogo.utils.argparsers import train_parser
from yogo.utils.cluster_anchors import best_anchor
from yogo.utils import draw_rects, get_wandb_confusion
from yogo.data.dataloader import (
    load_dataset_description,
    get_dataloader,
)


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def checkpoint_model(
    model: torch.nn.Module,
    epoch: int,
    name: str,
    step: int,
    normalized: bool,
    model_version: Optional[str] = None,
    **kwargs,
):
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "normalize_images": normalized,
            "model_state_dict": deepcopy(model.state_dict()),
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


def validate_process(
    halt_flag: mp.Event,
    param_q: mp.Queue,
    net,
    validate_dataloader,
    device,
    config,
    model_save_dir,
):
    while not halt_flag.is_set():
        try:
            params = param_q.get(timeout=1)
            param_q.task_done()
            param_q.put(params)
            global_step, epoch = 0,0
        except queue.Empty:
            continue
        net.load_state_dict(params)
        validate(
            net,
            validate_dataloader,
            device,
            config,
            global_step,
            model_save_dir,
            epoch,
        )


def validate(
    net,
    validate_dataloader,
    device,
    config,
    global_step,
    model_save_dir,
    epoch,
):
    classify = not config["no_classify"]
    class_names = config["class_names"]
    num_classes = len(class_names)
    # do validation things
    val_metrics = Metrics(
        num_classes=num_classes,
        device=device,
        class_names=class_names,
        classify=classify,
    )
    Y_loss = YOGOLoss(classify=classify).to(device)
    val_loss = 0.0
    net.eval()
    with torch.no_grad():
        for imgs, labels in validate_dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = net(imgs)
            loss = Y_loss(outputs, labels)
            val_loss += loss.item()
            val_metrics.update(outputs.detach(), labels.detach())

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

        mAP, confusion_data, precision, recall = val_metrics.compute()
        val_metrics.reset()

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
        if mAP["map"] > wandb.run.summary["best_mAP"]:
            wandb.run.summary["best_mAP"] = mAP["map"]
            wandb.log({"best_mAP_save": mAP["map"]}, step=global_step)
            checkpoint_model(
                net,
                epoch,
                model_save_dir / "best.pth",
                global_step,
                config["normalize_images"],
                model_version=config["model"],
            )
        else:
            checkpoint_model(
                net,
                epoch,
                model_save_dir / "latest.pth",
                global_step,
                config["normalize_images"],
                model_version=config["model"],
            )


def train():
    config = wandb.config
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    anchor_w = config["anchor_w"]
    anchor_h = config["anchor_h"]
    class_names = config["class_names"]
    weight_decay = config["weight_decay"]
    classify = not config["no_classify"]
    model = get_model_func(config["model"])
    num_classes = len(class_names)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        warnings.warn("no cuda devices found, using cpu - this will be slow!")
        device_list = ["cpu"]
    else:
        device_list = [f"cuda:{i}" for i in range(num_gpus)]

    wandb.config.update({"device_list": device_list})
    devices = cycle(device_list)
    print(device_list)

    device = next(devices)
    val_device = next(devices)

    test_metrics = Metrics(
        num_classes=num_classes,
        device=device,
        class_names=class_names,
        classify=classify,
    )

    net = None
    if config.pretrained_path:
        print(f"loading pretrained path from {config.pretrained_path}")

        net, net_cfg = YOGO.from_pth(config.pretrained_path)
        net.to(device)

        global_step = net_cfg["step"]
        config["normalize_images"] = net_cfg["normalize_images"]

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

    if num_gpus > 1:
        print("starting validation process")
        halt_flag = mp.Event()
        ctx = mp.get_context("spawn")
        param_q = ctx.Queue()
        val_proc = ctx.Process(
            target=validate_process,
            args=(
                halt_flag,
                param_q,
                net,
                validate_dataloader,
                val_device,
                config,
                model_save_dir,
            ),
            daemon=True,
        )
        val_proc.start()

    print("starting training")

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

        if num_gpus in (0, 1):
            validate(
                net,
                validate_dataloader,
                device,
                config,
                global_step,
                model_save_dir,
                epoch,
            )
        else:
            param_q.put(net.state_dict())
            param_q.get()
        net.train()

    halt_flag.set()

    net, cfg = YOGO.from_pth(model_save_dir / "best.pth")
    print(f"loaded best.pth from step {cfg['step']} for test inference")
    net.to(device)

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

    val_proc.join()


WandbConfig: TypeAlias = dict


def init_dataset(config: WandbConfig, Sx, Sy):
    dataloaders = get_dataloader(
        config["dataset_descriptor_file"],
        config["batch_size"],
        Sx=Sx,
        Sy=Sy,
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
            "training set size": f"{len(train_dataloader.dataset)} images",  # type:ignore
            "validation set size": f"{len(validate_dataloader.dataset)} images",  # type:ignore
            "testing set size": f"{len(test_dataloader.dataset)} images",  # type:ignore
        }
    )

    if wandb.run is not None:
        model_save_dir = Path(f"trained_models/{wandb.run.name}")
    else:
        model_save_dir = Path(
            f"trained_models/unnamed_run_{torch.randint(100, size=(1,)).item()}"
        )
    model_save_dir.mkdir(exist_ok=True, parents=True)

    return model_save_dir, train_dataloader, validate_dataloader, test_dataloader


def do_training(args) -> None:
    """responsible for parsing args and starting a training run"""
    epochs = args.epochs or df.EPOCHS
    batch_size = args.batch_size or df.BATCH_SIZE
    learning_rate = args.lr or df.LEARNING_RATE
    label_smoothing = args.label_smoothing or df.LABEL_SMOOTHING
    decay_factor = args.lr_decay_factor or df.DECAY_FACTOR
    weight_decay = args.weight_decay or df.WEIGHT_DECAY
    optimizer_type = args.optimizer or df.OPTIMIZER_TYPE

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
    dataset_paths = load_dataset_description(args.dataset_descriptor_file).dataset_paths

    print("getting best anchor")
    anchor_w, anchor_h = best_anchor([d["label_path"] for d in dataset_paths])

    print("initting wandb")
    wandb.init(
        project="yogo",
        entity="bioengineering",
        config={
            "optimizer_type": optimizer_type,
            "learning_rate": learning_rate,
            "decay_factor": decay_factor,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "epochs": epochs,
            "batch_size": batch_size,
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
    assert wandb.run is not None
    wandb.run.summary["best_mAP"] = 0

    train()


if __name__ == "__main__":
    parser = train_parser()
    args = parser.parse_args()

    do_training(args)
