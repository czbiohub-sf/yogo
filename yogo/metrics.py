import torch

from typing import Any, Tuple, List, Dict

from torchmetrics import MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix,
    MulticlassAccuracy,
    MulticlassROC,
    MulticlassCalibrationError,
)

from yogo.utils import get_wandb_confusion
from yogo.utils.prediction_formatting import (
    PredictionLabelMatch,
    format_preds_and_labels_v2,
)


class Metrics:
    @torch.no_grad()
    def __init__(
        self,
        classes: List[str],
        device: str = "cpu",
        sync_on_compute: bool = False,
        min_class_confidence_threshold: float = 0.9,
        include_mAP: bool = True,
        include_background: bool = True,
    ):
        self.device = device
        self.classes = classes + (["background"] if include_background else [])
        self.num_classes = len(classes)
        self.min_class_confidence_threshold = min_class_confidence_threshold
        self.include_mAP = include_mAP
        self.include_background = include_background

        # map can be very costly; so lets be able to turn it off if we
        # don't need it
        if include_mAP:
            self.mAP = MeanAveragePrecision(
                box_format="xyxy",
                sync_on_compute=sync_on_compute,
            )
            self.mAP.warn_on_many_detections = False
            self.mAP.to(device)

        self.confusion = MulticlassConfusionMatrix(
            num_classes=self.num_classes,
            validate_args=False,
            sync_on_compute=sync_on_compute,
        )
        self.confusion.to(device)

        self.prediction_metrics = MetricCollection(
            [
                MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average=None,
                    validate_args=False,
                    sync_on_compute=sync_on_compute,
                ),
                MulticlassROC(
                    num_classes=self.num_classes,
                    thresholds=500,
                    validate_args=False,
                    sync_on_compute=sync_on_compute,
                ),
                # per-class
                MulticlassPrecision(
                    num_classes=self.num_classes,
                    average=None,
                    validate_args=False,
                    sync_on_compute=sync_on_compute,
                ),
                # per-class
                MulticlassRecall(
                    num_classes=self.num_classes,
                    average=None,
                    validate_args=False,
                    sync_on_compute=sync_on_compute,
                ),
                MulticlassCalibrationError(
                    num_classes=self.num_classes,
                    n_bins=30,
                    validate_args=False,
                    sync_on_compute=sync_on_compute,
                ),
            ],
        )
        self.prediction_metrics.to(device)

        # We get some strange 'device-side assert error' when
        # calling unique() on a cuda tensor. So, we've explicitly
        # sent the relevant tensors to cpu. We've left the copy to
        # cpu explicit so we can easily come back later once we've
        # figured out the bug.

        # where YOGO misses an object
        self.num_obj_missed_by_class: torch.Tensor = torch.zeros(
            self.num_classes, dtype=torch.long, device="cpu"
        )
        # where YOGO predicts an object that isn't there
        self.num_obj_extra_by_class: torch.Tensor = torch.zeros(
            self.num_classes, dtype=torch.long, device="cpu"
        )
        self.total_num_true_objects = torch.zeros(1, dtype=torch.long, device="cpu")

    @torch.no_grad()
    def update(self, preds, labels, use_IoU: bool = True):
        bs, pred_shape, Sy, Sx = preds.shape
        bs, label_shape, Sy, Sx = labels.shape

        pred_label_matches: PredictionLabelMatch = PredictionLabelMatch.concat(
            [
                format_preds_and_labels_v2(
                    pred,
                    label,
                    min_class_confidence_threshold=self.min_class_confidence_threshold,
                )
                for pred, label in zip(preds.detach(), labels.detach())
            ]
        )

        def count_classes_in_tensor(class_predictions: torch.Tensor) -> torch.Tensor:
            values, unique_counts = class_predictions.unique(return_counts=True)
            final_counts = torch.zeros(self.num_classes, dtype=torch.long, device="cpu")
            final_counts[values.long()] = unique_counts
            return final_counts

        if pred_label_matches.missed_labels is not None:
            self.num_obj_missed_by_class += count_classes_in_tensor(
                pred_label_matches.missed_labels[:, 5].cpu()
            )

        if pred_label_matches.extra_predictions is not None:
            self.num_obj_extra_by_class += count_classes_in_tensor(
                pred_label_matches.extra_predictions[:, 5:].argmax(dim=1).cpu()
            )

        self.total_num_true_objects += pred_label_matches.labels.shape[0]

        if self.include_background:
            pred_label_matches = pred_label_matches.convert_background_errors(
                self.num_classes
            )

        fps, fls = pred_label_matches.preds, pred_label_matches.labels

        if self.include_mAP:
            self.mAP.update(*self._format_for_mAP(fps, fls))

        self.confusion.update(fps[:, 5:].argmax(dim=1), fls[:, 5:].squeeze())
        self.prediction_metrics.update(fps[:, 5:], fls[:, 5:].squeeze().long())

    @torch.no_grad()
    def compute(self) -> Tuple[Any, ...]:
        """We need to use rank here and exit early. The metrics sync on "compute",
        but then for steps after that, we only want to return if we're rank 0 (since
        that's synced to wandb).
        """
        pr_metrics = self.prediction_metrics.compute()

        if self.include_mAP:
            mAP_metrics = self.mAP.compute()
        else:
            mAP_metrics = {
                "map": torch.tensor(0.0),
            }

        confusion_metrics = self.confusion.compute()

        return (
            mAP_metrics,
            confusion_metrics,
            pr_metrics["MulticlassAccuracy"],
            pr_metrics["MulticlassROC"],
            pr_metrics["MulticlassPrecision"],
            pr_metrics["MulticlassRecall"],
            pr_metrics["MulticlassCalibrationError"].item(),
            self.num_obj_missed_by_class.cpu(),
            self.num_obj_extra_by_class.cpu(),
            self.total_num_true_objects.cpu(),
        )

    @torch.no_grad()
    def get_wandb_confusion_matrix(self, confusion_metrics):
        return get_wandb_confusion(
            confusion_metrics, self.classes, "test confusion matrix"
        )

    def reset(self):
        if self.include_mAP:
            self.mAP.reset()
        self.confusion.reset()
        self.prediction_metrics.reset()

    @torch.no_grad()
    def forward(self, preds, labels):
        self.update(preds, labels)
        res = self.compute()
        self.reset()
        return res

    def _format_for_mAP(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        """
        formatted_preds
           tensor of predictions shape=[N, x y x y objectness *classes]
        formatted_labels
           tensor of labels shape=[N, mask x y x y class])
        """
        formatted_preds, formatted_labels = [], []

        for fp, fl in zip(preds, labels):
            formatted_preds.append(
                {
                    "boxes": fp[:4].reshape(1, 4),
                    "scores": fp[4].reshape(1),
                    "labels": fp[5:].argmax().reshape(1),
                }
            )
            formatted_labels.append(
                {
                    "boxes": fl[1:5].reshape(1, 4),
                    "labels": fl[5].reshape(1).long(),
                }
            )

        return formatted_preds, formatted_labels
