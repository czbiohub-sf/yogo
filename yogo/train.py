from __future__ import annotations

import os
import sys
import wandb
import torch
import warnings

from pathlib import Path
from copy import deepcopy
from typing_extensions import TypeAlias
from typing import Any, Tuple, Optional, Collection, Union

import torch.multiprocessing as mp

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP

from yogo.model import YOGO
from yogo.metrics import Metrics
from yogo.data.yogo_dataloader import get_dataloader
from yogo.data.dataset_definition_file import DatasetDefinition
from yogo.yogo_loss import YOGOLoss
from yogo.model_defns import get_model_func
from yogo.utils.argparsers import train_parser
from yogo.utils.default_hyperparams import DefaultHyperparams as df
from yogo.utils import (
    draw_yogo_prediction,
    get_wandb_roc,
    get_free_port,
    choose_device,
)


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


WandbConfig: TypeAlias = dict


class Trainer:
    """
    Simple trainer class. `train_from_ddp` is the main entry point to training, though
    almost entirely, use the CLI to train.
    """

    def __init__(
        self,
        config: WandbConfig,
        _rank: int = 0,
        _world_size: int = 1,
    ) -> None:
        self.config = config

        self.device = f"cuda:{_rank}"

        self._rank = _rank
        self._world_size = _world_size

        self.Sx: Optional[int] = None
        self.Sy: Optional[int] = None
        self.model_save_dir: Optional[Path] = None
        self.dataset_definition: Optional[DatasetDefinition] = None

        self.epoch = 0
        self.global_step = 0
        self.min_val_loss = float("inf")

        self._initialized = False

    @classmethod
    def train_from_ddp(
        cls, _rank: int, _world_size: int, config: WandbConfig
    ) -> Trainer:
        """
        Due to mp spawn not giving a kwarg option, we have to give `rank` first. But, for
        the sake of consistency, we would like to give the config first instead. A bit messy.
        """
        trainer = cls(config, _rank=_rank, _world_size=_world_size)
        trainer.init()
        trainer.train()
        return trainer

    def init(self) -> None:
        self._init_tcp_store()
        self._init_dataset_definition()
        self._init_model()
        self._init_dataset()
        self._init_training_tools()
        self._init_wandb()
        self._initialized = True

    def _init_tcp_store(self) -> None:
        os.environ["YOGO_TCP_STORE_PORT"] = self.config["tcp_store_port"]

        # store for distributed training (so far just using it for model_save_dir)
        self._store = torch.distributed.TCPStore(
            "localhost",
            int(os.environ["YOGO_TCP_STORE_PORT"]),
            self._world_size,  # number of clients
            self._rank == 0,  # only rank 0 should be the server, others are clients
        )

    def _init_dataset_definition(self) -> None:
        """
        we need to initialize this separately because we need
        the number of classes for the dataset definition for model
        initialization, but also need Sx, Sy for dataset initialization.
        So we pull this out and execute it first.
        """
        self.dataset_definition = DatasetDefinition.from_yaml(
            Path(self.config["dataset_descriptor_file"])
        )
        self.config["class_names"] = self.dataset_definition.classes

    def _init_model(self) -> None:
        if self.dataset_definition is None:
            raise RuntimeError("dataset definition not initialized")

        if (
            self.config["pretrained_path"] is None
            or self.config["pretrained_path"] == "none"
        ):
            net = YOGO(
                img_size=self.config["image_hw"],
                anchor_w=self.config["anchor_w"],
                anchor_h=self.config["anchor_h"],
                is_rgb=self.config["rgb"],
                num_classes=len(self.config["class_names"]),
                model_func=get_model_func(self.config["model"]),
            ).to(self.device)
            self.global_step = 0
        else:
            net, net_cfg = YOGO.from_pth(self.config["pretrained_path"])

            if any(net.img_size.cpu().numpy() != self.config["image_hw"]):
                raise RuntimeError(
                    "mismatch in pretrained network image resize shape and current resize shape: "
                    f"pretrained network image_hw = {net.img_size}, requested image_hw = {self.config['image_hw']}"
                )

            net.to(self.device)
            self.global_step = net_cfg["step"]
            self.config["normalize_images"] = net.normalize_images
            self.config["model"] = net.model_version

        self.Sx, self.Sy = net.get_grid_size()

        os.environ["MASTER_ADDR"] = "0.0.0.0"
        os.environ["MASTER_PORT"] = self.config["master_port"]

        torch.distributed.init_process_group(
            backend="nccl", rank=self._rank, world_size=self._world_size
        )

        self.net = DDP(net, device_ids=[self._rank])

    def _init_dataset(self) -> None:
        if self.Sx is None or self.Sy is None:
            raise RuntimeError("model not initialized")
        elif self.dataset_definition is None:
            raise RuntimeError("dataset definition not initialized")

        dataloaders = get_dataloader(
            self.dataset_definition,
            self.config["batch_size"],
            Sx=self.Sx,
            Sy=self.Sy,
            image_hw=self.config["image_hw"],
            rgb=self.config["rgb"],
            normalize_images=self.config["normalize_images"],
            split_fraction_override=self.config["dataset_split_override"],
        )

        train_dataloader = dataloaders["train"]
        # sneaky hack to replace non-existant datasets with empty list
        validate_dataloader: Union[DataLoader[Any], Collection] = dataloaders.get(
            "val", []
        )
        test_dataloader: Union[DataLoader[Any], Collection] = dataloaders.get(
            "test", []
        )

        if self._dataset_size(validate_dataloader) == 0:
            warnings.warn("no validation dataset found")

        if self._dataset_size(test_dataloader) == 0:
            warnings.warn("no test dataset found")

        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.test_dataloader = test_dataloader

    @staticmethod
    def _dataset_size(dataloader: Union[Collection, DataLoader]) -> int:
        # type ignore for dataset-sized type error
        return (
            len(dataloader.dataset)  # type: ignore
            if isinstance(dataloader, DataLoader)
            else len(dataloader)
        )

    def _init_training_tools(self) -> None:
        self.Y_loss = YOGOLoss(
            no_obj_weight=self.config["no_obj_weight"],
            iou_weight=self.config["iou_weight"],
            label_smoothing=self.config["label_smoothing"],
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

    def _init_wandb(self) -> None:
        if self._rank != 0:
            return

        run_id = wandb.util.generate_id()
        wandb.init(
            id=run_id,
            config=self.config,
            entity=self.config["wandb_entity"],
            project=self.config["wandb_project"],
            name=self.config["name"],
            notes=self.config["note"],
            tags=self.config["tags"],
        )

        wandb.config.update(
            {
                "Sx": self.Sx,
                "Sy": self.Sy,
                "training set size": f"{self._dataset_size(self.train_dataloader)} images",  # type:ignore
                "validation set size": f"{self._dataset_size(self.validate_dataloader)} images",  # type:ignore
                "testing set size": f"{self._dataset_size(self.test_dataloader)} images",  # type:ignore
                "normalize_images": self.config["normalize_images"],
                "wandb_run_id": run_id,
            },
            allow_val_change=True,
        )

        trained_model_dir = Path(f"{__file__}").parent.parent / "trained_models"
        if wandb.run is not None:
            model_save_dir = trained_model_dir / wandb.run.name
        else:
            model_save_dir = (
                trained_model_dir
                / f"run_{torch.randint(100000000, size=(1,)).item():08}"
            )

        model_save_dir.mkdir(exist_ok=True, parents=True)
        self.model_save_dir = model_save_dir

        self._store.set("model_save_dir", str(model_save_dir.resolve()))

    def checkpoint(
        self,
        filename: Union[str, Path],
        model_name: str,
        **kwargs,
    ) -> None:
        if isinstance(self.net, DDP):
            state_dict = self.net.module.state_dict()
            model_version = self.net.module.model_version
        else:
            state_dict = self.net.state_dict()
            model_version = self.net.model_version

        torch.save(
            {
                "epoch": self.epoch,
                "step": self.global_step,
                "normalize_images": self.config["normalize_images"],
                "classes": self.config["class_names"],
                "model_name": model_name,
                "model_state_dict": deepcopy(state_dict),
                "optimizer_state_dict": deepcopy(self.optimizer.state_dict()),
                "model_version": model_version,
                **kwargs,
            },
            str(filename),
        )

    def train(self) -> None:
        torch.distributed.barrier()

        if not self._initialized:
            raise RuntimeError("trainer not initialized")

        device = self.device

        for epoch in range(self.config["epochs"]):
            self.epoch = epoch
            # mypy thinks that self.train_dataloader has type Iterable[Any]?
            self.train_dataloader.sampler.set_epoch(epoch)  # type: ignore

            self.net.train()
            for imgs, labels in self.train_dataloader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(
                    dtype=torch.float16,
                    enabled=self.config["half"],
                ):
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

            if epoch % 4 == 0:
                self._validate()

        model_save_dir = Path(self._store.get("model_save_dir").decode("utf-8"))

        if (model_save_dir / "best.pth").exists():
            model_checkpoint = torch.load(
                model_save_dir / "best.pth", map_location="cpu"
            )
            self.net.module.load_state_dict(model_checkpoint["model_state_dict"])
        else:
            warnings.warn(
                f"no best model found at {model_save_dir / 'best.pth'} for testing..."
            )

        test_metrics = self.test(
            self.test_dataloader,
            self.device,
            self.config,
            self.net,
        )

        if self._rank == 0 and test_metrics is not None:
            self._log_test_metrics(*test_metrics)
        elif self._rank == 0 and test_metrics is None:
            warnings.warn(
                "no test metrics found - most likely test_dataloader is empty"
            )

        wandb.finish()

        torch.distributed.destroy_process_group()

    @torch.no_grad()
    def _validate(self) -> None:
        if self._dataset_size(self.validate_dataloader) == 0:
            return

        net_state = self.net.training
        self.net.eval()
        device = self.device

        val_loss = torch.tensor(0.0, device=device)
        # TODO figure out correct type for dataloader
        for imgs, labels in self.validate_dataloader:  # type: ignore
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(
                dtype=torch.float16,
                enabled=self.config["half"],
            ):
                outputs = self.net(imgs)
                loss, _ = self.Y_loss(outputs, labels)

            val_loss += loss

        # TODO mypy thinks that ReduceOp doesn't have AVG;
        # maybe because AVG is only available for nccl backend? How to address?
        torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG)  # type: ignore

        # back to training!
        self.net.train(net_state)

        if self._rank != 0:
            return

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

        self.model_save_dir = Path(self._store.get("model_save_dir").decode("utf-8"))
        if mean_val_loss < self.min_val_loss:
            self.min_val_loss = mean_val_loss
            wandb.log({"best_val_loss": mean_val_loss}, step=self.global_step)
            self.checkpoint(
                self.model_save_dir / "best.pth",
                model_name=(
                    wandb.run.name if wandb.run is not None else "recent_run_best"
                ),
            )
        else:
            self.checkpoint(
                self.model_save_dir / "latest.pth",
                model_name=(
                    wandb.run.name if wandb.run is not None else "recent_run_latest"
                ),
            )

    @staticmethod
    @torch.no_grad()
    def test(
        test_dataloader: Union[Collection, DataLoader],
        device: Union[str, torch.device],
        config: WandbConfig,
        net: torch.nn.Module,
        rank: int = 0,
        include_mAP: bool = True,
        include_background: bool = False,
    ) -> Optional[Tuple[Any, ...]]:
        if Trainer._dataset_size(test_dataloader) == 0:
            return None

        net_state = net.training
        net.eval()

        Trainer._check_keys(config)

        test_metrics = Metrics(
            classes=config["class_names"],
            device=str(device),
            sync_on_compute=False,
            include_mAP=include_mAP,
            include_background=include_background,
        )

        Y_loss = YOGOLoss(
            no_obj_weight=config["no_obj_weight"],
            iou_weight=config["iou_weight"],
            label_smoothing=config["label_smoothing"],
        ).to(device)

        test_loss = torch.zeros(1, device=device)

        for imgs, labels in test_dataloader:
            imgs = imgs.to(device, non_blocking=False)
            labels = labels.to(device, non_blocking=False)

            with torch.cuda.amp.autocast(
                dtype=torch.float16,
                enabled=config["half"],
            ):
                outputs = net(imgs)
                loss, _ = Y_loss(outputs, labels)

            test_loss += loss
            test_metrics.update(outputs.detach(), labels.detach())

        mean_loss = test_loss / len(test_dataloader)  # type: ignore

        (
            mAP,
            confusion_data,
            accuracy,
            roc_curves,
            precision,
            recall,
            calibration_error,
            num_obj_missed_by_class,
            num_obj_extra_by_class,
            total_num_true_objects,
        ) = test_metrics.compute()

        net.train(net_state)

        if rank != 0:
            return None

        return (
            mean_loss.item(),  # type: ignore
            mAP,
            test_metrics.get_wandb_confusion_matrix(confusion_data),
            accuracy,
            roc_curves,
            precision,
            recall,
            calibration_error,
            num_obj_missed_by_class,
            num_obj_extra_by_class,
            total_num_true_objects,
            config["class_names"],
        )

    @staticmethod
    def _check_keys(config):
        required_test_keys = (
            "class_names",
            "iou_weight",
            "no_obj_weight",
            "label_smoothing",
            "half",
        )

        for key in required_test_keys:
            if key not in config:
                raise ValueError(
                    f"{key} is required in config (full list of keys: {required_test_keys})"
                )

    @staticmethod
    def _log_test_metrics(
        mean_test_loss,
        mAP,
        confusion_data,
        accuracy,
        roc_curves,
        precision,
        recall,
        calibration_error,
        num_obj_missed_by_class,
        num_obj_extra_by_class,
        total_num_true_objects,
        class_names,
    ):
        """
        kind-of a crummy, hacky method to log everything to W&B. Not pretty, but functional.
        Functional as in "it works", not as in "pure function"
        """
        accuracy_table = wandb.Table(
            data=[[labl, acc] for labl, acc in zip(class_names, accuracy)],
            columns=["label", "accuracy"],
        )

        fpr, tpr, thresholds = roc_curves

        wandb.summary["test loss"] = mean_test_loss
        wandb.summary["test mAP"] = mAP["map"]
        wandb.summary["test mAP (full)"] = mAP
        wandb.summary["test precision"] = precision.mean()
        wandb.summary["test recall"] = recall.mean()
        wandb.summary["calibration error"] = calibration_error
        wandb.summary["num obj missed by class"] = num_obj_missed_by_class
        wandb.summary["num obj extra by class"] = num_obj_extra_by_class
        wandb.summary["total num true objects"] = total_num_true_objects

        per_class_precision, per_class_recall = dict(), dict()
        for i, cn in enumerate(class_names):
            per_class_precision[f"test precision {cn}"] = precision[i].item()
            per_class_recall[f"test recall {cn}"] = recall[i].item()

        wandb.summary["per-class precision"] = per_class_precision
        wandb.summary["per-class recall"] = per_class_recall

        wandb.log(
            {
                "test confusion": confusion_data,
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
    device = torch.device(args.device) if args.device is not None else choose_device()

    anchor_w, anchor_h = df.ANCHOR_W, df.ANCHOR_H

    config = {
        "learning_rate": args.learning_rate,
        "decay_factor": args.lr_decay_factor,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "iou_weight": args.iou_weight,
        "no_obj_weight": args.no_obj_weight,
        "classify_weight": args.classify_weight,
        "tcp_store_port": str(get_free_port()),
        "master_port": str(get_free_port()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
        "anchor_w": anchor_w,
        "anchor_h": anchor_h,
        "model": args.model,
        "half": args.half,
        "rgb": args.rgb_images,
        "image_hw": args.image_hw,
        "pretrained_path": args.from_pretrained,
        "normalize_images": args.normalize_images,
        "dataset_split_override": args.dataset_split_override,
        "dataset_descriptor_file": args.dataset_descriptor_file,
        "slurm-job-id": os.getenv("SLURM_JOB_ID", default=None),
        "torch-version": torch.__version__,
        "python-version": sys.version,
        "name": args.name,
        "note": args.note,
        "tags": args.tags,
        "wandb_entity": args.wandb_entity,
        "wandb_project": args.wandb_project,
    }

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError(
            "at least 1 gpu is required for training; if cpu training "
            "is required, we can add it back"
        )

    wandb.login(anonymous="allow")

    mp.spawn(
        Trainer.train_from_ddp, args=(world_size, config), nprocs=world_size, join=True
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = train_parser()
    args = parser.parse_args()

    do_training(args)
