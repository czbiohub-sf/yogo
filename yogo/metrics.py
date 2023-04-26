import torch

import torchvision.ops as ops

from typing import Optional, Tuple, List, Dict

from torchmetrics import MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix,
    # MulticlassAccuracy,
)


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
        self.precision_recall_metrics = MetricCollection(
            [
                MulticlassPrecision(num_classes=num_classes, thresholds=4),
                MulticlassRecall(num_classes=num_classes, thresholds=4),
                # MulticlassAccuracy(num_classes=num_classes, thresholds=4)
            ]
        )

        self.mAP.to(device)
        self.confusion.to(device)
        self.precision_recall_metrics.to(device)

        self.num_classes = num_classes
        self.class_names = (
            list(range(num_classes)) if class_names is None else class_names
        )
        self.classify = classify
        assert self.num_classes == len(self.class_names)

    def update(self, preds, labels):
        bs, Sy, Sx, pred_shape = preds.shape
        bs, Sy, Sx, label_shape = labels.shape

        formatted_preds, formatted_labels = self._format_preds_and_labels(
            preds, labels, use_IoU=True, per_batch=True
        )

        self.mAP.update(*self.format_for_mAP(formatted_preds, formatted_labels))

        formatted_preds = torch.cat(formatted_preds)
        formatted_labels = torch.cat(formatted_labels)

        self.confusion.update(
            formatted_preds[:, 5:].argmax(dim=1), formatted_labels[:, 5:].squeeze()
        )

        self.precision_recall_metrics.update(
            formatted_preds[:, 5:], formatted_labels[:, 5:].squeeze().long()
        )

    def compute(self):
        pr_metrics = self.precision_recall_metrics.compute()
        return (
            self.mAP.compute(),
            self.confusion.compute(),
            pr_metrics["MulticlassPrecision"],
            pr_metrics["MulticlassRecall"],
            # pr_metrics["MulticlassAccuracy"],
        )

    def reset(self):
        self.mAP.reset()
        self.confusion.reset()
        self.precision_recall_metrics.reset()

    def forward(self, preds, labels):
        self.update(preds, labels)
        res = self.compute()
        self.reset()
        return res

    def _format_preds_and_labels(
        self,
        pred_batch: torch.Tensor,
        label_batch: torch.Tensor,
        use_IoU: bool = True,
        objectness_thresh: float = 0.3,
        per_batch: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A very important utility function for filtering predictions on labels

        Often, we need to calculate conditional probabilites - e.g. #(correct predictions | objectness > thresh)
        We want to select our predicted bbs and class predictions on IOU, and sometimes on ojbectness, e.t.c.

        This is also a very slow function right now (due to `mask`s, which require cudaMemcpyAsync or cudaStreamSynchronize,
        depending on implementation).

        - pred_batch and label_batch are the batch label and prediction tensors, hot n' fresh from the model and dataloader!
        - use_IoU is whether to use IoU instead of naive cell matching. More accurate, but slower.
        - objectness_thresh is the "objectness" threshold, YOGO's confidence that there is a prediction in the given cell. Can
            only be used with use_IoU == True

        returns
            (
                tensor of predictions shape=[N, x y x y objectness *classes],
                tensor of labels shape=[N, mask x y x y class]
            )
        """
        if not (0 <= objectness_thresh < 1):
            raise ValueError(
                f"must have 0 <= objectness_thresh < 1; got objectness_thresh={objectness_thresh}"
            )

        (
            bs1,
            Sy,
            Sx,
            pred_shape,
        ) = pred_batch.shape  # pred_shape is xc yc w h objectness *classes
        (
            bs2,
            Sy,
            Sx,
            label_shape,
        ) = label_batch.shape  # label_shape is mask x y x y class
        assert bs1 == bs2, "sanity check, pred batch size should be equal"

        formatted_preds = pred_batch.view(bs1 * Sx * Sy, pred_shape)
        formatted_labels = label_batch.view(bs2 * Sx * Sy, label_shape)

        formatted_preds[:, 0:4] = ops.box_convert(
            formatted_preds[:, 0:4], "cxcywh", "xyxy"
        )

        labels_mask = formatted_labels[:, 0].bool()
        labels_mask_sum = labels_mask.view(bs2, -1).sum(dim=1)

        objectness_mask = (formatted_preds[:, 4] > objectness_thresh).bool()
        objectness_mask_sum = objectness_mask.view(bs1, -1).sum(dim=1)

        masked_predictions, masked_labels = [], []

        for b in range(bs1):
            mini, maxi = b * Sx * Sy, (b + 1) * Sx * Sy

            labels = formatted_labels[mini:maxi][labels_mask[mini:maxi], :]

            preds_with_objects_by_labels = formatted_preds[mini:maxi][
                labels_mask[mini:maxi], :
            ]
            preds_with_objects = formatted_preds[mini:maxi][
                objectness_mask[mini:maxi], :
            ]

            if use_IoU and objectness_mask_sum[b] >= labels_mask_sum[b]:
                # choose predictions from argmaxed IoU along label dim to get best prediction per label
                prediction_matrix = ops.box_iou(
                    labels[:, 1:5],
                    preds_with_objects[:, 0:4],
                )
                n, m = prediction_matrix.shape
                if m > 0:
                    # add mini here so we take correct slice
                    prediction_indices = prediction_matrix.argmax(dim=1)
                else:
                    # no predictions!
                    prediction_indices = []

                final_preds = preds_with_objects[prediction_indices]
            else:
                # filter on label tensor idx
                final_preds = preds_with_objects_by_labels

            masked_predictions.append(final_preds)
            masked_labels.append(labels)

        if per_batch:
            return masked_predictions, masked_labels
        return torch.cat(masked_predictions), torch.cat(masked_labels)

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
