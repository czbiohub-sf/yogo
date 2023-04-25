#! /usr/bin/env python3

import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from pathlib import Path
from typing_extensions import TypeAlias
from typing import Optional, Tuple, cast

from yogo.model import YOGO
from yogo.yogo_loss import YOGOLoss
from yogo.metrics import Metrics
from yogo.data.dataloader import (
    load_dataset_description,
    get_dataloader,
)
from yogo.utils.cluster_anchors import best_anchor
from yogo.utils.argparsers import train_parser


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


def trace_handler(prof):
    prof.export_chrome_trace("chrome_profile.json")


def train(config):
    device = config["device"]
    anchor_w = config["anchor_w"]
    anchor_h = config["anchor_h"]
    class_names = config["class_names"]
    num_classes = len(class_names)

    net = YOGO(
        num_classes=num_classes,
        img_size=config["resize_shape"],
        anchor_w=anchor_w,
        anchor_h=anchor_h,
    ).to(device)
    Y_loss = YOGOLoss().to(device)

    optimizer = AdamW(net.parameters(), lr=config["learning_rate"])

    metrics = Metrics(num_classes=num_classes, device=device, class_names=class_names)

    # TODO: generalize so we can tune Sx / Sy!
    # TODO: best way to make model architecture tunable?
    Sx, Sy = net.get_grid_size()

    (
        model_save_dir,
        train_dataloader,
        validate_dataloader,
        test_dataloader,
    ) = init_dataset(config, Sx, Sy)

    min_period = 8 * len(train_dataloader)
    anneal_period = config["epochs"] * len(train_dataloader) - min_period

    lin = LinearLR(optimizer, start_factor=0.01, end_factor=1, total_iters=min_period)
    cs = CosineAnnealingLR(optimizer, T_max=anneal_period, eta_min=5e-5)
    scheduler = SequentialLR(optimizer, [lin, cs], [min_period])

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=8,
            active=54,
        ),
        with_stack=True,
        on_trace_ready=trace_handler,
    ) as p:
        for i, (imgs, labels, label_idxs) in enumerate(train_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            label_idxs = label_idxs.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = net(imgs)
            loss = Y_loss(outputs, labels)
            loss.backward()

            optimizer.step()
            scheduler.step()
            metrics.update(outputs.detach(), labels, label_idxs)

            p.step()

            if i > 64:
                break


WandbConfig: TypeAlias = dict


def init_dataset(config, Sx, Sy):
    dataloaders = get_dataloader(
        config["dataset_descriptor_file"],
        config["batch_size"],
        Sx,
        Sy,
        device=config["device"],
        preprocess_type=config["preprocess_type"],
        vertical_crop_size=config["vertical_crop_size"],
        resize_shape=config["resize_shape"],
    )

    train_dataloader = dataloaders["train"]
    validate_dataloader = dataloaders["val"]
    test_dataloader = dataloaders["test"]

    model_save_dir = Path(
        f"trained_models/unnamed_run_{torch.randint(100, size=(1,)).item()}"
    )
    model_save_dir.mkdir(exist_ok=True, parents=True)

    return model_save_dir, train_dataloader, validate_dataloader, test_dataloader


def do_training(args) -> None:
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    epochs = args.epochs or 64
    batch_size = args.batch_size or 32

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

    desc = load_dataset_description(args.dataset_descriptor_file)
    class_names, dataset_paths = desc.classes, desc.dataset_paths

    anchor_w, anchor_h = best_anchor([d["label_path"] for d in dataset_paths])

    config = {
        "learning_rate": 1e-6,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "anchor_w": anchor_w,
        "anchor_h": anchor_h,
        "resize_shape": resize_target_size,
        "vertical_crop_size": vertical_crop_size,
        "preprocess_type": preprocess_type,
        "class_names": class_names,
        "run group": args.group,
        "dataset_descriptor_file": args.dataset_descriptor_file,
    }

    train(config)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")

    parser = train_parser()
    args = parser.parse_args()

    do_training(args)
