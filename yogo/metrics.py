import torch


from typing import Optional, Tuple, List, Dict

from torchmetrics import MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix,
    MulticlassAccuracy,
    MulticlassROC,
)

from yogo.utils.utils import format_preds_and_labels


class Metrics:
    @torch.no_grad()
    def __init__(
        self,
        num_classes: int,
        device: str = "cpu",
        class_names: Optional[List[str]] = None,
        classify: bool = True,
    ):
        # TODO can we put confusion in MetricCollection? mAP?
        self.mAP = MeanAveragePrecision(box_format="xyxy")
        self.confusion = MulticlassConfusionMatrix(num_classes=num_classes)
        self.prediction_metrics = MetricCollection(
            [
                MulticlassPrecision(num_classes=num_classes, thresholds=None),
                MulticlassRecall(num_classes=num_classes, thresholds=None),
                MulticlassAccuracy(
                    num_classes=num_classes, thresholds=None, average=None
                ),
                MulticlassROC(num_classes=num_classes, thresholds=500, average=None),
            ]
        )

        self.mAP.to(device)
        self.confusion.to(device)
        self.prediction_metrics.to(device)

        self.num_classes = num_classes
        self.class_names: List[str] = (
            [str(n) for n in range(num_classes)] if class_names is None else class_names
        )
        self.classify = classify
        assert self.class_names is not None and self.num_classes == len(
            self.class_names
        )

    def update(self, preds, labels, use_IoU: bool = True):
        bs, pred_shape, Sy, Sx = preds.shape
        bs, label_shape, Sy, Sx = labels.shape

        formatted_preds, formatted_labels = zip(
            *[
                format_preds_and_labels(pred, label, use_IoU=use_IoU)
                for pred, label in zip(preds, labels)
            ]
        )

        self.mAP.update(*self.format_for_mAP(formatted_preds, formatted_labels))

        fps, fls = torch.cat(formatted_preds), torch.cat(formatted_labels)

        self.confusion.update(fps[:, 5:].argmax(dim=1), fls[:, 5:].squeeze())

        self.prediction_metrics.update(fps[:, 5:], fls[:, 5:].squeeze().long())

    def compute(self):
        pr_metrics = self.prediction_metrics.compute()
        return (
            self.mAP.compute(),
            self.confusion.compute(),
            pr_metrics["MulticlassPrecision"],
            pr_metrics["MulticlassRecall"],
            pr_metrics["MulticlassAccuracy"],
            pr_metrics["MulticlassROC"],
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

    def format_for_mAP(
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
                    "labels": fp[:, 5:].argmax(dim=1) if self.classify else fl[:, 5],
                }
            )
            labels.append(
                {
                    "boxes": fl[:, 1:5],
                    "labels": fl[:, 5],
                }
            )

        return preds, labels
