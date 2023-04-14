import torch

import torchvision.ops as ops

from typing import Optional, Tuple, List, Dict

from torchmetrics import ConfusionMatrix
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall


class Metrics:
    @torch.no_grad()
    def __init__(
        self,
        num_classes: int,
        device: str = "cpu",
        class_names: Optional[List[str]] = None,
        classify: bool = True,
    ):
        self.mAP = MeanAveragePrecision(box_format="xyxy")
        self.confusion = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        # TODO review https://torchmetrics.readthedocs.io/en/stable/classification/precision.html
        self.precision = MulticlassPrecision(num_classes=num_classes, thresholds=4)
        self.recall = MulticlassRecall(num_classes=num_classes, thresholds=4)

        self.mAP.to(device)
        self.confusion.to(device)
        self.precision.to(device)
        self.recall.to(device)

        self.num_classes = num_classes
        self.class_names = (
            list(range(num_classes)) if class_names is None else class_names
        )
        self.classify = classify
        assert self.num_classes == len(self.class_names)

    def update(self, preds, labels):
        bs, pred_shape, Sy, Sx = preds.shape
        bs, label_shape, Sy, Sx = labels.shape

        self.mAP.update(*self.format_for_mAP(preds, labels))

        formatted_preds, formatted_labels = self._format_preds_and_labels(
            preds, labels, use_IoU=True
        )

        self.confusion.update(
            formatted_preds[:, 5:].argmax(dim=1), formatted_labels[:, 5:].squeeze()
        )

        self.precision.update(
            formatted_preds[:, 5:], formatted_labels[:, 5:].squeeze().long()
        )

        self.recall.update(
            formatted_preds[:, 5:], formatted_labels[:, 5:].squeeze().long()
        )

    def compute(self):
        return (
            self.mAP.compute(),
            self.confusion.compute(),
            self.precision.compute(),
            self.recall.compute(),
        )

    def reset(self):
        self.mAP.reset()
        self.confusion.reset()
        self.precision.reset()
        self.recall.reset()

    def forward(self, preds, labels):
        # prob inefficient but its OK
        self.update(preds, labels)
        res = self.compute()
        self.reset()
        return res

    def _format_preds_and_labels(
        self,
        batch_preds: torch.Tensor,
        batch_labels: torch.Tensor,
        use_IoU: bool = True,
        objectness_thresh: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A very important utility function for filtering predictions on labels

        Often, we need to calculate conditional probabilites - e.g. #(correct predictions | objectness > thresh)
        We want to select our predicted bbs and class predictions on IOU, and sometimes on ojbectness, e.t.c

        batch_preds and batch_labels are the batch label and prediction tensors, hot n' fresh from the model and dataloader!
        use_IoU is whether to use IoU instead of naive cell matching. More accurate, but slower.
        objectness_thresh is the "objectness" threshold, YOGO's confidence that there is a prediction in the given cell. Can
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
            pred_shape,
            Sy,
            Sx,
        ) = batch_preds.shape  # pred_shape is xc yc w h objectness *classes
        (
            bs2,
            label_shape,
            Sy,
            Sx,
        ) = batch_labels.shape  # label_shape is mask x y x y class
        assert bs1 == bs2, "sanity check, pred batch size should equal"

        masked_predictions, masked_labels = [], []
        for b in range(bs1):
            reformatted_preds = batch_preds[b, ...].view(pred_shape, Sx * Sy).T
            reformatted_labels = batch_labels[b, ...].view(label_shape, Sx * Sy).T

            # reformatted_labels[:, 0] = 1 if there is a label for that cell, else 0
            labels_mask = reformatted_labels[:, 0].bool()
            objectness_mask = (reformatted_preds[:, 4] > objectness_thresh).bool()

            img_masked_labels = reformatted_labels[labels_mask]

            if use_IoU and objectness_mask.sum() >= len(img_masked_labels):
                # filter on objectness
                preds_with_objects = reformatted_preds[objectness_mask]

                preds_with_objects[:, 0:4] = ops.box_convert(
                    preds_with_objects[:, 0:4], "cxcywh", "xyxy"
                )

                # choose predictions from argmaxed IoU along label dim to get best prediction per label
                prediction_matrix = ops.box_iou(
                    img_masked_labels[:, 1:5], preds_with_objects[:, 0:4]
                )
                n, m = prediction_matrix.shape
                if m > 0:
                    prediction_indices = prediction_matrix.argmax(dim=1)
                else:
                    # no predictions!
                    prediction_indices = []
                final_preds = preds_with_objects[prediction_indices]
            else:
                # filter on label tensor idx
                final_preds = reformatted_preds[reformatted_labels[:, 0].bool()]
                final_preds[:, 0:4] = ops.box_convert(
                    final_preds[:, 0:4], "cxcywh", "xyxy"
                )

            masked_predictions.append(final_preds)
            masked_labels.append(img_masked_labels)

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
        preds = [
            {
                "boxes": formatted_preds[:, :4],
                "scores": formatted_preds[:, 4],
                "labels": (
                    formatted_preds[:, 5:].argmax(dim=1)
                    if self.classify
                    else formatted_labels[:, 5]
                ),
            }
        ]
        labels = [
            {
                "boxes": formatted_labels[:, 1:5],
                "labels": formatted_labels[:, 5],
            }
        ]

        return preds, labels
