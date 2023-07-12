#! /usr/bin/env python3

import os
import wandb
import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from pathlib import Path
from copy import deepcopy
from typing_extensions import TypeAlias
from typing import Optional, cast, Iterable


from yogo.model import YOGO
from yogo.metrics import Metrics
from yogo.data import YOGO_CLASS_ORDERING
from yogo.data.yogo_dataloader import get_dataloader, normalized_inverse_frequencies
from yogo.data.dataset_description_file import load_dataset_description
from yogo.yogo_loss import YOGOLoss
from yogo.model_defns import get_model_func
from yogo.utils.argparsers import train_parser
from yogo.utils.cluster_anchors import best_anchor
from yogo.utils import (
    draw_yogo_prediction,
    get_wandb_confusion,
    get_wandb_roc,
    Timer,
)


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


WandbConfig: TypeAlias = dict


def checkpoint_model(
    model: torch.nn.Module,
    epoch: int,
    optimizer: torch.nn.Module,
    filename: str,
    step: int,
    normalized: bool,
    model_name: str,
    model_version: Optional[str] = None,
    **kwargs,
):
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "normalize_images": normalized,
            "model_name": model_name,
            "model_state_dict": deepcopy(model.state_dict()),
            "optimizer_state_dict": deepcopy(optimizer.state_dict()),
            "model_version": model_version,
            **kwargs,
        },
        str(filename),
    )


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
    # sneaky hack to replace non-existant datasets with emtpy list
    validate_dataloader: Iterable = dataloaders.get("val", [])
    test_dataloader: Iterable = dataloaders.get("test", [])

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

    test_metrics = Metrics(
        num_classes=num_classes,
        device=device,
        class_names=class_names,
        classify=classify,
    )

    if config.pretrained_path is None or config.pretrained_path == "none":
        net = YOGO(
            num_classes=num_classes,
            img_size=config["resize_shape"],
            anchor_w=anchor_w,
            anchor_h=anchor_h,
            model_func=model,
        ).to(device)
        global_step = 0
    else:
        print(f"loading pretrained path from {config.pretrained_path}")

        net, net_cfg = YOGO.from_pth(config.pretrained_path)
        net.to(device)

        global_step = net_cfg["step"]
        wandb.config.update(
            {"normalize_images": net_cfg["normalize_images"]}, allow_val_change=True
        )

        if any(net.img_size.cpu().numpy() != config["resize_shape"]):
            raise RuntimeError(
                "mismatch in pretrained network image resize shape and current resize shape: "
                f"pretrained network resize_shape = {net.img_size}, requested resize_shape = {config['resize_shape']}"
            )

    Sx, Sy = net.get_grid_size()
    wandb.config.update({"Sx": Sx, "Sy": Sy})

    (
        model_save_dir,
        train_dataloader,
        validate_dataloader,
        test_dataloader,
    ) = init_dataset(config, Sx, Sy)

    class_weights = normalized_inverse_frequencies(
        [
            config["healthy_weight"],
            config["ring_weight"],
            config["troph_weight"],
            config["schizont_weight"],
            config["gametocyte_weight"],
            config["wbc_weight"],
            config["misc_weight"],
        ]
    )
    wandb.config.update({"class_weights": class_weights.tolist()})

    Y_loss = YOGOLoss(
        no_obj_weight=config["no_obj_weight"],
        iou_weight=config["iou_weight"],
        classify_weight=config["classify_weight"],
        label_smoothing=config["label_smoothing"],
        class_weights=class_weights,
        temperature=config["logit_norm_temperature"],
        classify=classify,
    ).to(device)

    optimizer = AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_dataloader),
        eta_min=learning_rate / config["decay_factor"],
    )

    min_val_loss = float("inf")
    for epoch in range(epochs):
        # train
        for imgs, labels in train_dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = net(imgs)
            loss, loss_components = Y_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            wandb.log(
                {
                    "train loss": loss.item(),
                    "epoch": epoch,
                    "LR": scheduler.get_last_lr()[0],
                    "training grad norm": net.grad_norm(),
                    "training param norm": net.param_norm(),
                    **loss_components,
                },
                commit=global_step % 100 == 0,
                step=global_step,
            )

        # do validation things
        val_loss = torch.tensor(0.0, device=device)
        net.eval()
        with torch.no_grad():
            for imgs, labels in validate_dataloader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = net(imgs)
                loss, _ = Y_loss(outputs, labels)
                val_loss += loss

            # just use the final imgs and labels for val!
            annotated_img = wandb.Image(
                draw_yogo_prediction(
                    imgs[0, ...],
                    outputs[0, ...].detach(),
                    labels=class_names,
                    images_are_normalized=config["normalize_images"],
                )
            )
            mean_val_loss = val_loss.item() / len(validate_dataloader)
            wandb.log(
                {
                    "validation bbs": annotated_img,
                    "val loss": mean_val_loss,
                },
                step=global_step,
            )

            if mean_val_loss < min_val_loss:
                min_val_loss = mean_val_loss
                wandb.log({"best_val_loss": mean_val_loss}, step=global_step)
                checkpoint_model(
                    net,
                    epoch,
                    optimizer,
                    model_save_dir / "best.pth",
                    global_step,
                    config["normalize_images"],
                    model_name=wandb.run.name,
                    model_version=config["model"],
                )
            else:
                checkpoint_model(
                    net,
                    epoch,
                    optimizer,
                    model_save_dir / "latest.pth",
                    global_step,
                    config["normalize_images"],
                    model_name=wandb.run.name,
                    model_version=config["model"],
                )

        net.train()

    net, cfg = YOGO.from_pth(model_save_dir / "best.pth")
    print(f"loaded best.pth from step {cfg['step']} for test inference")
    net.to(device)

    test_loss = 0.0
    with torch.no_grad():
        for imgs, labels in test_dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = net(imgs)
            loss, _ = Y_loss(outputs, labels)
            test_loss += loss.item()
            test_metrics.update(outputs.detach(), labels.detach())

        (
            mAP,
            confusion_data,
            precision,
            recall,
            accuracy,
            roc_curves,
        ) = test_metrics.compute()
        test_metrics.reset()

        accuracy_table = wandb.Table(
            data=[[labl, acc] for labl, acc in zip(class_names, accuracy)],
            columns=["label", "accuracy"],
        )

        fpr, tpr, thresholds = roc_curves

        wandb.summary["test loss"] = test_loss / len(test_dataloader)
        wandb.summary["test mAP"] = mAP["map"]
        wandb.summary["test precision"] = precision
        wandb.summary["test recall"] = recall
        wandb.log(
            {
                "test confusion": get_wandb_confusion(
                    confusion_data, class_names, "test confusion matrix"
                ),
                "test accuracy": wandb.plot.bar(
                    accuracy_table, "label", "accuracy", title="test accuracy"
                ),
                "test ROC": get_wandb_roc(
                    fpr=[t.tolist() for t in fpr],
                    tpr=[t.tolist() for t in tpr],
                    thresholds=[t.tolist() for t in thresholds],
                    classes=class_names,
                ),
            }
        )


def do_training(args) -> None:
    """responsible for parsing args and starting a training run"""
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    preprocess_type: Optional[str]
    vertical_crop_size: Optional[float] = None

    if args.crop_height:
        vertical_crop_size = cast(float, args.crop_height)
        if not (0 < vertical_crop_size < 1):
            raise ValueError(
                "vertical_crop_size must be between 0 and 1; got {vertical_crop_size}"
            )
        resize_target_size = (round(vertical_crop_size * 772), 1032)
        preprocess_type = "crop"
    else:
        resize_target_size = (772, 1032)
        preprocess_type = None

    with Timer("loading dataset description"):
        dataset_paths = load_dataset_description(
            args.dataset_descriptor_file
        ).dataset_paths

    with Timer("getting best anchor"):
        anchor_w, anchor_h = best_anchor([d["label_path"] for d in dataset_paths])

    with Timer("initting wandb"):
        wandb.init(
            project="yogo",
            entity="bioengineering",
            config={
                "learning_rate": args.learning_rate,
                "decay_factor": args.lr_decay_factor,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "iou_weight": args.iou_weight,
                "no_obj_weight": args.no_obj_weight,
                "classify_weight": args.classify_weight,
                # class weights
                "healthy_weight": 8,
                "ring_weight": 2,
                "troph_weight": 1,
                "schizont_weight": 1,
                "gametocyte_weight": 1,
                "wbc_weight": 2,
                "misc_weight": 2,
                # end class weights
                "logit_norm_temperature": args.logit_norm_temperature,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
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
                "normalize_images": args.normalize_images,
                "dataset_descriptor_file": args.dataset_descriptor_file,
                "slurm-job-id": os.getenv("SLURM_JOB_ID", default=None),
            },
            name=args.name,
            notes=args.note,
            tags=(args.tag,) if args.tag is not None else None,
        )

    train()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = train_parser()
    args = parser.parse_args()

    do_training(args)
