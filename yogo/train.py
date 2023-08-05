import os
import wandb
import torch

from pathlib import Path
from copy import deepcopy
from typing_extensions import TypeAlias
from typing import Optional, cast, Iterable

import torch.multiprocessing as mp

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP

from yogo.model import YOGO
from yogo.metrics import Metrics
from yogo.data import YOGO_CLASS_ORDERING
from yogo.data.yogo_dataloader import get_dataloader
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


class Trainer:
    def __init__(
        self, config: WandbConfig, _rank: int = 0, _world_size: int = 1
    ) -> None:
        self.config = config

        self.device = f"cuda:{_rank}"

        self._rank = _rank
        self._world_size = _world_size

        self.Sx: Optional[int] = None
        self.Sy: Optional[int] = None

        self.epoch = 0
        self.global_step = 0
        self.min_val_loss = float("inf")

        self._initialized = True

    @classmethod
    def train_from_ddp(cls, _rank: int, _world_size: int, config: WandbConfig) -> None:
        """
        Due to mp spawn not giving a kwarg option, we have to give `rank` first. But, for
        the sake of consistency, we want to give the config first. A bit messy.
        """
        trainer = cls(config, _rank=_rank, _world_size=_world_size)
        trainer.init()
        trainer.train()

    def init(self) -> None:
        self._init_model()
        self._init_dataset()
        self._init_training_tools()
        self._init_wandb()
        self._initialized = True

    def _init_model(self) -> None:
        if (
            self.config["pretrained_path"] is None
            or self.config["pretrained_path"] == "none"
        ):
            net = YOGO(
                num_classes=len(self.config["class_names"]),
                img_size=self.config["resize_shape"],
                anchor_w=self.config["anchor_w"],
                anchor_h=self.config["anchor_h"],
                model_func=get_model_func(self.config["model"]),
            ).to(self.device)
            self.global_step = 0
        else:
            net, net_cfg = YOGO.from_pth(self.config["pretrained_path"])
            if any(net.img_size.cpu().numpy() != self.config["resize_shape"]):
                raise RuntimeError(
                    "mismatch in pretrained network image resize shape and current resize shape: "
                    f"pretrained network resize_shape = {net.img_size}, requested resize_shape = {self.config['resize_shape']}"
                )
            net.to(self.device)
            self.global_step = net_cfg["step"]
            self.config["normalize_images"] = net_cfg["normalize_images"]

        self.Sx, self.Sy = net.get_grid_size()

        if self._world_size > 1:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"

            torch.distributed.init_process_group(
                backend="nccl", rank=self._rank, world_size=self._world_size
            )

            net = DDP(net, device_ids=[self._rank])

        self.net = net

    def _init_dataset(self) -> None:
        if self.Sx is None or self.Sy is None:
            raise RuntimeError("model not initialized")

        dataloaders = get_dataloader(
            self.config["dataset_descriptor_file"],
            self.config["batch_size"],
            Sx=self.Sx,
            Sy=self.Sy,
            preprocess_type=self.config["preprocess_type"],
            vertical_crop_size=self.config["vertical_crop_size"],
            resize_shape=self.config["resize_shape"],
            normalize_images=self.config["normalize_images"],
            rank=self._rank,
            world_size=self._world_size,
        )

        train_dataloader = dataloaders["train"]
        # sneaky hack to replace non-existant datasets with emtpy list
        validate_dataloader: Iterable = dataloaders.get("val", [])
        test_dataloader: Iterable = dataloaders.get("test", [])

        if wandb.run is not None:
            model_save_dir = Path(f"trained_models/{wandb.run.name}")
        else:
            model_save_dir = Path(
                f"trained_models/unnamed_run_{torch.randint(100, size=(1,)).item()}"
            )
        model_save_dir.mkdir(exist_ok=True, parents=True)

        self.model_save_dir = model_save_dir
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.test_dataloader = test_dataloader

    def _init_training_tools(self):
        class_weights = [self.config["healthy_weight"], 1, 1, 1, 1, 1, 1]

        self.Y_loss = YOGOLoss(
            no_obj_weight=self.config["no_obj_weight"],
            iou_weight=self.config["iou_weight"],
            label_smoothing=self.config["label_smoothing"],
            class_weights=torch.tensor(class_weights),
            logit_norm_temperature=self.config["logit_norm_temperature"],
            classify=not self.config["no_classify"],
        ).to(self.device)

        self.optimizer = AdamW(
            self.net.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["epochs"] * len(self.train_dataloader),
            eta_min=self.config["learning_rate"] / self.config["decay_factor"],
        )

    def _init_wandb(self):
        if self._rank != 0:
            return

        wandb.init(
            project="yogo",
            entity="bioengineering",
            config=self.config,
            name=self.config["name"],
            notes=self.config["note"],
            tags=(self.config["tag"],) if self.config["tag"] is not None else None,
        )

        wandb.watch(self.net)
        wandb.config.update(
            {
                "Sx": self.Sx,
                "Sy": self.Sy,
                "training set size": f"{len(self.train_dataloader.dataset)} images",  # type:ignore
                "validation set size": f"{len(self.validate_dataloader.dataset)} images",  # type:ignore
                "testing set size": f"{len(self.test_dataloader.dataset)} images",  # type:ignore
                "normalize_images": self.config["normalize_images"],
            },
            allow_val_change=True,
        )

    def checkpoint(
        self,
        filename: str,
        model_name: str,
        model_version: Optional[str] = None,
        **kwargs,
    ):
        if isinstance(self.net, DDP):
            pass

        torch.save(
            {
                "epoch": self.epoch,
                "step": self.global_step,
                "normalize_images": self.config["normalize_images"],
                "model_name": model_name,
                "model_state_dict": deepcopy(self.net.state_dict()),
                "optimizer_state_dict": deepcopy(self.optimizer.state_dict()),
                "model_version": model_version,
                **kwargs,
            },
            str(filename),
        )

    def train(self):
        if not self._initialized:
            raise RuntimeError("trainer not initialized")

        device = self.device

        for epoch in range(self.config["epochs"]):
            self.epoch = epoch
            self.train_dataloader.sampler.set_epoch(epoch)

            for imgs, labels in self.train_dataloader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                outputs = self.net(imgs)

                loss, loss_components = self.Y_loss(outputs, labels)
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                self.global_step += 1

                if self._rank == 0:
                    wandb.log(
                        {
                            "train loss": loss.item(),
                            "epoch": epoch,
                            "LR": self.scheduler.get_last_lr()[0],
                            **loss_components,
                        },
                        commit=self.global_step % 100 == 0,
                        step=self.global_step,
                    )

            if self._rank == 0:
                self._validate()

        if self._rank == 0:
            self._test()

    @torch.no_grad()
    def _validate(self):
        self.net.eval()
        device = self.device

        val_loss = torch.tensor(0.0, device=device)
        for imgs, labels in self.validate_dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = self.net(imgs)

            loss, _ = self.Y_loss(outputs, labels)
            val_loss += loss

        # just use the final imgs and labels for val!
        annotated_img = wandb.Image(
            draw_yogo_prediction(
                imgs[0, ...],
                outputs[0, ...].detach(),
                labels=self.config["class_names"],
                images_are_normalized=self.config["normalize_images"],
            )
        )

        mean_val_loss = val_loss.item() / len(self.validate_dataloader)

        wandb.log(
            {
                "validation bbs": annotated_img,
                "val loss": mean_val_loss,
            },
            step=self.global_step,
        )

        if mean_val_loss < self.min_val_loss:
            self.min_val_loss = mean_val_loss
            wandb.log({"best_val_loss": mean_val_loss}, step=self.global_step)
            self.checkpoint(
                self.model_save_dir / "best.pth",
                model_name=wandb.run.name,
                model_version=self.config["model"],
            )
        else:
            self.checkpoint(
                self.model_save_dir / "latest.pth",
                model_name=wandb.run.name,
                model_version=self.config["model"],
            )

        self.net.train()

    @torch.no_grad()
    def _test(self):
        """
        TODO could make this static so we can evaluate YOGO
        separately from training
        """
        device = self.device

        test_metrics = Metrics(
            class_names=self.config["class_names"],
            classify=not self.config["no_classify"],
            device=device,
        )

        net, cfg = YOGO.from_pth(self.model_save_dir / "best.pth")
        net.to(device)

        test_loss = 0.0
        for imgs, labels in self.test_dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = self.net(imgs)
            loss, _ = self.Y_loss(outputs, labels)

            test_loss += loss.item()
            test_metrics.update(outputs.detach(), labels.detach())

        (
            mAP,
            confusion_data,
            precision,
            recall,
            accuracy,
            roc_curves,
            calibration_error,
        ) = test_metrics.compute()
        test_metrics.reset()

        accuracy_table = wandb.Table(
            data=[
                [labl, acc] for labl, acc in zip(self.config["class_names"], accuracy)
            ],
            columns=["label", "accuracy"],
        )

        fpr, tpr, thresholds = roc_curves

        wandb.summary["test loss"] = test_loss / len(self.test_dataloader)
        wandb.summary["test mAP"] = mAP["map"]
        wandb.summary["test precision"] = precision
        wandb.summary["test recall"] = recall
        wandb.summary["calibration error"] = calibration_error

        wandb.log(
            {
                "test confusion": get_wandb_confusion(
                    confusion_data, self.config["class_names"], "test confusion matrix"
                ),
                "test accuracy": wandb.plot.bar(
                    accuracy_table, "label", "accuracy", title="test accuracy"
                ),
                "test ROC": get_wandb_roc(
                    fpr=[t.tolist() for t in fpr],
                    tpr=[t.tolist() for t in tpr],
                    thresholds=[t.tolist() for t in thresholds],
                    classes=self.config["class_names"],
                ),
            }
        )
        wandb.finish()
        torch.distributed.destroy_process_group()


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

    config = {
        "learning_rate": args.learning_rate,
        "decay_factor": args.lr_decay_factor,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "iou_weight": args.iou_weight,
        "no_obj_weight": args.no_obj_weight,
        "classify_weight": args.classify_weight,
        "healthy_weight": args.healthy_weight,
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
        "name": args.name,
        "note": args.note,
        "tag": args.tag,
    }

    world_size = torch.cuda.device_count()

    mp.spawn(
        Trainer.train_from_ddp, args=(world_size, config), nprocs=world_size, join=True
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = train_parser()
    args = parser.parse_args()

    do_training(args)
