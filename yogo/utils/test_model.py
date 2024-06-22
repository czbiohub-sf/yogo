#! /usr/bin/env python3

import os
import torch
import wandb
import pickle
import argparse
import warnings

from torch.utils.data import Dataset, DataLoader

from yogo.model import YOGO
from yogo.train import Trainer
from yogo.data.yogo_dataset import ObjectDetectionDataset
from yogo.data.utils import collate_batch_robust
from yogo.data.dataset_definition_file import DatasetDefinition
from yogo.data.yogo_dataloader import (
    get_dataloader,
    choose_dataloader_num_workers,
)


def test_model(args: argparse.Namespace) -> None:
    device = "cuda"

    y, cfg = YOGO.from_pth(args.pth_path, inference=False)
    y.to(device)

    data_defn = DatasetDefinition.from_yaml(args.dataset_defn_path)

    config = {
        "class_names": data_defn.classes,
        "no_classify": False,
        "iou_weight": 1,
        "no_obj_weight": 0.5,
        "label_smoothing": 0.0001,
        "half": True,
        "model": args.pth_path,
        "test_set": args.dataset_defn_path,
        "slurm-job-id": os.getenv("SLURM_JOB_ID", default=None),
    }

    log_to_wandb = args.wandb or (args.wandb_resume_id is not None)

    if log_to_wandb:
        print("logging to wandb")
        wandb.init(
            config=config,
            entity=args.wandb_entity,
            project=args.wandb_project,
            id=args.wandb_resume_id,
            resume="allow",
            tags=args.tags,
            notes=args.note,
        )

        if (wandb.run is not None) and wandb.run.offline:
            warnings.warn(
                "wandb run is offline - will not be logged "
                "to wandb.ai but to the local disc"
            )

        assert wandb.run is not None
        wandb.run.tags += type(wandb.run.tags)(["resumed for test"])

    dataloaders = get_dataloader(
        data_defn,
        64,
        y.get_grid_size()[0],
        y.get_grid_size()[1],
        normalize_images=cfg["normalize_images"],
    )

    test_dataset: Dataset[ObjectDetectionDataset] = dataloaders["test"].dataset
    num_workers = choose_dataloader_num_workers(len(test_dataset))  # type: ignore

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        batch_size=64,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(111111),
        collate_fn=collate_batch_robust,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

    test_metrics = Trainer.test(
        test_dataloader,
        device,
        config,
        y,
        include_mAP=args.include_mAP,
        include_background=args.include_background,
    )

    if log_to_wandb:
        Trainer._log_test_metrics(*test_metrics)  # type: ignore

    if args.dump_to_disk:
        pickle.dump(test_metrics, open("test_metrics.pkl", "wb"))


def do_model_test(args):
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError(
            "at least 1 gpu is required for testing (otherwise it's painfully slow); "
            "if cpu testing is required, we can add it back"
        )

    wandb.login(anonymous="allow")

    test_model(args)
