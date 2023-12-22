import torch

from typing import Tuple, List, Dict

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

from yogo.utils import format_preds_and_labels


class Metrics:
    @torch.no_grad()
    def __init__(
        self,
        num_classes: int,
        device: str = "cpu",
        classify: bool = True,
        sync_on_compute: bool = False,
        min_class_confidence_threshold: float = 0.9,
    ):
        self.num_classes = num_classes
        self.classify = classify
        self.min_class_confidence_threshold = min_class_confidence_threshold

        self.mAP = MeanAveragePrecision(
            box_format="xyxy",
            backend="faster_coco_eval",
            sync_on_compute=sync_on_compute,
        )
        self.mAP.warn_on_many_detections = False

        self.confusion = MulticlassConfusionMatrix(
            num_classes=self.num_classes,
            validate_args=False,
            sync_on_compute=sync_on_compute,
        )
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
                MulticlassPrecision(
                    num_classes=self.num_classes,
                    validate_args=False,
                    sync_on_compute=sync_on_compute,
                ),
                MulticlassRecall(
                    num_classes=self.num_classes,
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

        self.mAP.to(device)
        self.confusion.to(device)
        self.prediction_metrics.to(device)

    def update(self, preds, labels, use_IoU: bool = True):
        bs, pred_shape, Sy, Sx = preds.shape
        bs, label_shape, Sy, Sx = labels.shape

        formatted_preds, formatted_labels = zip(
            *[
                format_preds_and_labels(
                    pred,
                    label,
                    use_IoU=use_IoU,
                    min_class_confidence_threshold=self.min_class_confidence_threshold,
                )
                for pred, label in zip(preds, labels)
            ]
        )

        self.mAP.update(*self._format_for_mAP(formatted_preds, formatted_labels))

        fps, fls = torch.cat(formatted_preds), torch.cat(formatted_labels)

        self.confusion.update(fps[:, 5:].argmax(dim=1), fls[:, 5:].squeeze())

        self.prediction_metrics.update(fps[:, 5:], fls[:, 5:].squeeze().long())

    def compute(self):
        pr_metrics = self.prediction_metrics.compute()

        mAP_metrics = self.mAP.compute()

        confusion_metrics = self.confusion.compute()

        return (
            mAP_metrics,
            confusion_metrics,
            pr_metrics["MulticlassAccuracy"],
            pr_metrics["MulticlassROC"],
            pr_metrics["MulticlassPrecision"],
            pr_metrics["MulticlassRecall"],
            pr_metrics["MulticlassCalibrationError"].item(),
        )

    def reset(self):
        self.mAP.reset()
        self.confusion.reset()
        self.prediction_metrics.reset()

    def forward(self, preds, labels):
        self.update(preds, labels)
        res = self.compute()
        self.reset()
        return res

    def _format_for_mAP(
        self, formatted_preds, formatted_labels
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        """
        formatted_preds
           tensor of predictions shape=[N, x y x y objectness *classes]
        formatted_labels
           tensor of labels shape=[N, mask x y x y class])
        """
        preds, labels = [], []
        for fp, fl in zip(formatted_preds, formatted_labels):
            preds.append(
                {
                    "boxes": fp[:, :4],
                    "scores": fp[:, 4],
                    "labels": fp[:, 5:].argmax(dim=1)
                    if self.classify
                    else fl[:, 5].long(),
                }
            )
            labels.append(
                {
                    "boxes": fl[:, 1:5],
                    "labels": fl[:, 5].long(),
                }
            )

        return preds, labels
