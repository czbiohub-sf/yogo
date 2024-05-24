import torch
import warnings
import numpy as np
import numpy.typing as npt

import torchvision.ops as ops

from dataclasses import dataclass
from typing import (
    List,
    Tuple,
    Literal,
    Optional,
    get_args,
)

from scipy.optimize import linear_sum_assignment


BoxFormat = Literal["xyxy", "cxcywh"]


def format_preds(
    pred: torch.Tensor,
    obj_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    box_format: BoxFormat = "cxcywh",
    min_class_confidence_threshold: float = 0.0,
) -> torch.Tensor:
    """
    formats pred, prediction tensor straight from YOGO, into [N,pred_shape], after applying NMS,
    and, thresholding objectness, and filtering thin boxes. box_format specifies the returned box format.

    An OK lower bound is 1e-6

    For all thresholds, set to 0 to disable.

    Parameters
    ----------
    pred: torch.Tensor
        Raw YOGO output (unbatched)
    obj_thresh: float = 0.5
        Objectness threshold
    iou_thresh: float = 0.5
        Intersection over union threshold (for non-maximal suppression (NMS))
    box_format: BoxFormat = 'cxcywh'
        Bounding box format, defaults to (center x, center y, width, height). Can also be (top left x, top left y, bottom right x, bottom right y)
    min_class_confidence_threshold: float = 0.0
        filters out all predictions with a maximum confidence less than this threshold
    """

    if len(pred.shape) != 3:
        raise ValueError(
            "argument to format_pred should be unbatched result - "
            f"shape should be (pred_shape, Sy, Sx), got {pred.shape}"
        )
    elif box_format not in get_args(BoxFormat):
        raise ValueError(
            f"invalid box format {box_format}; valid box formats are {get_args(BoxFormat)}"
        )

    pred_shape, Sy, Sx = pred.shape

    reformatted_preds = pred.view(pred_shape, Sx * Sy).T

    # Filter for objectness first
    objectness_mask = (reformatted_preds[:, 4] > obj_thresh).bool()
    preds = reformatted_preds[objectness_mask]

    # if we have to convert box format to xyxy, do it to the tensor
    # and give nms a view of the original. Otherwise, just give nms
    # the a converted clone of the boxes.
    if box_format == "xyxy":
        preds[:, :4] = ops.box_convert(preds[:, :4], "cxcywh", "xyxy")
        nms_boxes = preds[:, :4]
    elif box_format == "cxcywh":
        nms_boxes = ops.box_convert(preds[:, :4], "cxcywh", "xyxy")

    # Non-maximal supression to remove duplicate boxes
    if iou_thresh > 0:
        keep_idxs = ops.nms(
            nms_boxes,
            torch.max(preds[:, 5:], dim=1).values * preds[:, 4],
            iou_threshold=iou_thresh,
        )
        preds = preds[keep_idxs]

    # Filter out predictions with low class confidence
    if min_class_confidence_threshold > 0:
        keep_idxs = preds[:, 5:].max(dim=1).values > min_class_confidence_threshold
        preds = preds[keep_idxs]

    return preds


def format_to_numpy(
    img_id: int,
    prediction_tensor: np.ndarray,
    img_h: int,
    img_w: int,
    np_dtype=np.float32,
) -> npt.NDArray:
    """Function to parse a prediction tensor and save it in a numpy format

    Parameters
    ----------
    prediction_tensor: np.ndarray
        The direct output tensor from a call to the YOGO model (1 * (5+NUM_CLASSES) * (Sx*Sy))
    img_h: int
    img_w: int
    np_dtype: np.dtype

    Returns
    -------
    npt.NDArray: (15 x N):
        0 img_ids (1 x N)
        1 top left x (1 x N)
        2 top right y (1 x N)
        3 bottom right x (1 x N)
        4 bottom right y (1 x N)
        5 objectness (1 x N)
        6 peak pred_labels (1 x N)
        7 peak pred_probs (1 x N)
        8-14 pred_probs (NUM_CLASSES x N)

    Where the width of the array (N) is the total number of objects detected
    in the dataset.
    """

    filtered_pred = (
        format_preds(
            torch.from_numpy(prediction_tensor),
            box_format="xyxy",
        )
        .numpy()
        .T
    )

    img_ids = np.ones(filtered_pred.shape[1]).astype(np_dtype) * img_id
    tlx = filtered_pred[0, :] * img_w
    tly = filtered_pred[1, :] * img_h
    brx = filtered_pred[2, :] * img_w
    bry = filtered_pred[3, :] * img_h

    objectness = filtered_pred[4, :].astype(np_dtype)
    all_confs = filtered_pred[5:, :].astype(np_dtype)

    pred_labels = np.argmax(all_confs, axis=0).astype(np.uint8)
    pred_probs = filtered_pred[5:,][pred_labels, np.arange(filtered_pred.shape[1])]

    pred_labels = pred_labels.astype(np_dtype)
    pred_probs = pred_probs.astype(np_dtype)

    return np.vstack(
        (img_ids, tlx, tly, brx, bry, objectness, pred_labels, pred_probs, all_confs)
    )


def one_hot(idx, num_classes):
    return torch.nn.functional.one_hot(
        torch.tensor(idx, dtype=torch.long), num_classes=num_classes
    )


@dataclass
class PredictionLabelMatch:
    """
    When matching object detection predictions to labels, we have three
    cases to consider:
        1 there is a one-to-one match between predictions and labels
        2 some predictions are actually background
        3 some labels are missed
    we want to represent these nicely. This is a little dataclass to represent
    these cases, specifically for format_preds_and_labels_v2.
    """

    preds: torch.Tensor
    labels: torch.Tensor
    missed_labels: Optional[torch.Tensor]
    extra_predictions: Optional[torch.Tensor]

    @staticmethod
    def concat(
        preds_and_labels: List["PredictionLabelMatch"],
    ) -> "PredictionLabelMatch":

        missed_labels_ = [
            p.missed_labels for p in preds_and_labels if p.missed_labels is not None
        ]
        extra_predictions_ = [
            p.extra_predictions
            for p in preds_and_labels
            if p.extra_predictions is not None
        ]
        return PredictionLabelMatch(
            preds=torch.cat([p.preds for p in preds_and_labels]),
            labels=torch.cat([p.labels for p in preds_and_labels]),
            missed_labels=(
                torch.cat(missed_labels_, dim=0) if missed_labels_ else None
            ),
            extra_predictions=(
                torch.cat(extra_predictions_, dim=0) if extra_predictions_ else None
            ),
        )

    def convert_background_errors(self, num_classes: int) -> "PredictionLabelMatch":
        """
        Assumes that the ``background'' class is the last class
        TODO right now we convert to list and back for mypy, but that's a bad tradeoff
        and really we just need to figure out how to properly type this with torch
        """
        new_preds, new_labels = [], []

        missed_labels = (
            [] if self.missed_labels is None else self.missed_labels.tolist()
        )
        extra_predictions = (
            [] if self.extra_predictions is None else self.extra_predictions.tolist()
        )

        for missed_label in missed_labels:
            new_preds.append(
                torch.tensor(
                    [
                        *missed_label[1:5],
                        1,
                        *one_hot(num_classes - 1, num_classes).float(),
                    ]
                )
            )
            new_labels.append(torch.tensor(missed_label))

        for extra_prediction in extra_predictions:
            new_preds.append(torch.tensor([*extra_prediction, 0]))  # add background
            new_labels.append(torch.tensor([1, *extra_prediction[:4], num_classes - 1]))

        new_preds_ten = torch.stack(new_preds).to(self.preds.device)
        new_labels_ten = torch.stack(new_labels).to(self.labels.device)

        # add background class to end of self.preds too
        self.preds = torch.cat(
            [self.preds, torch.zeros(self.preds.shape[0], 1, device=self.preds.device)],
            dim=1,
        )

        return PredictionLabelMatch(
            preds=torch.cat([self.preds, new_preds_ten]),
            labels=torch.cat([self.labels, new_labels_ten]),
            missed_labels=None,
            extra_predictions=None,
        )


def format_preds_and_labels_v2(
    pred: torch.Tensor,
    label: torch.Tensor,
    objectness_thresh: float = 0.5,
    min_class_confidence_threshold: float = 0.0,
) -> PredictionLabelMatch:
    """A very important utility function for filtering predictions on labels

    Often, we need to calculate conditional probabilites - e.g. #(correct predictions | objectness > thresh)
    We want to select our predicted bbs and class predictions on IOU, and sometimes on ojbectness, e.t.c

    preds and labels are the batch label and prediction tensors, hot n' fresh from the model and dataloader!
    use_IoU is whether to use IoU instead of naive cell matching. More accurate, but slower.
    objectness_thresh is the "objectness" threshold, YOGO's confidence that there is a prediction in the given cell. Can
    only be used with use_IoU == True
    """
    pred.squeeze_()
    label.squeeze_()

    if len(pred.shape) != 3:
        raise ValueError(
            "argument to format_pred should be unbatched result - "
            f"shape should be (pred_shape, Sy, Sx), got {pred.shape}"
        )

    formatted_preds = format_preds(
        pred,
        obj_thresh=objectness_thresh,
        iou_thresh=0.5,
        box_format="xyxy",
        min_class_confidence_threshold=min_class_confidence_threshold,
    )

    (
        label_shape,
        Sy,
        Sx,
    ) = label.shape  # label_shape is mask x y x y class
    labels = label.view(label_shape, Sx * Sy).T
    formatted_labels = labels[labels[:, 0].bool()]

    M, _ = formatted_preds.shape
    N, _ = formatted_labels.shape
    pairwise_iou = ops.box_iou(formatted_labels[:, 1:5], formatted_preds[:, :4])

    cost_matrix = 1 - pairwise_iou.cpu().numpy()
    row_idxs, col_idxs = linear_sum_assignment(cost_matrix)

    row_idxs = torch.tensor(row_idxs, dtype=torch.long)
    col_idxs = torch.tensor(col_idxs, dtype=torch.long)

    # Matched predictions and labels
    matched_preds = formatted_preds[col_idxs]
    matched_labels = formatted_labels[row_idxs]

    all_pred_indices = torch.arange(M, dtype=torch.long)
    unmatched_pred_indices = torch.tensor(
        [i for i in all_pred_indices if i not in col_idxs],
        dtype=torch.long,
        device=formatted_preds.device,
    )
    extra_preds = formatted_preds[unmatched_pred_indices]

    all_label_indices = torch.arange(N, dtype=torch.long)
    unmatched_label_indices = torch.tensor(
        [i for i in all_label_indices if i not in row_idxs],
        dtype=torch.long,
        device=formatted_labels.device,
    )
    missed_labels = formatted_labels[unmatched_label_indices]

    return PredictionLabelMatch(
        preds=matched_preds,
        labels=matched_labels,
        missed_labels=missed_labels,
        extra_predictions=extra_preds,
    )


def format_preds_and_labels(
    pred: torch.Tensor,
    label: torch.Tensor,
    use_IoU: bool = True,
    objectness_thresh: float = 0.5,
    min_class_confidence_threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A very important utility function for filtering predictions on labels

    Often, we need to calculate conditional probabilites - e.g. #(correct predictions | objectness > thresh)
    We want to select our predicted bbs and class predictions on IOU, and sometimes on ojbectness, e.t.c

    preds and labels are the batch label and prediction tensors, hot n' fresh from the model and dataloader!
    use_IoU is whether to use IoU instead of naive cell matching. More accurate, but slower.
    objectness_thresh is the "objectness" threshold, YOGO's confidence that there is a prediction in the given cell. Can
        only be used with use_IoU == True

    returns
        tuple(
            tensor of predictions shape=[N, x y x y objectness *classes],
            tensor of labels shape=[N, mask x y x y class]
        )
    """
    warnings.warn("use format_preds_and_labels_v2 instead", DeprecationWarning)

    pred.squeeze_()
    label.squeeze_()

    if len(pred.shape) != 3:
        raise ValueError(
            "argument to format_pred should be unbatched result - "
            f"shape should be (pred_shape, Sy, Sx), got {pred.shape}"
        )

    if not (0 <= objectness_thresh < 1):
        raise ValueError(
            f"must have 0 <= objectness_thresh < 1; got objectness_thresh={objectness_thresh}"
        )

    (
        pred_shape,
        Sy,
        Sx,
    ) = pred.shape  # pred_shape is xc yc w h objectness *classes
    (
        label_shape,
        Sy,
        Sx,
    ) = label.shape  # label_shape is mask x y x y class

    reformatted_preds = pred.view(pred_shape, Sx * Sy).T  # [N, pred_shape]
    reformatted_labels = label.view(label_shape, Sx * Sy).T  # [N, label_shape]

    # reformatted_labels[:, 0] = 1 if there is a label for that cell, else 0
    objectness_mask = (reformatted_preds[:, 4] > objectness_thresh).bool()

    # calculate_the_confidence_mask
    values, _ = torch.max(reformatted_preds[:, 5:], dim=1)
    class_confidence_mask = (values > min_class_confidence_threshold).bool()

    # the total prediction mask is where confidence + objectness are high
    # by default, though, min_class_confidence is 0 so it's just objectness
    pred_mask = class_confidence_mask & objectness_mask

    labels_mask = reformatted_labels[:, 0].bool()
    labels_with_objects = reformatted_labels[labels_mask]

    if use_IoU and pred_mask.sum() >= len(labels_with_objects):
        # filter on objectness
        preds_with_objects = reformatted_preds[pred_mask]

        preds_with_objects[:, 0:4] = ops.box_convert(
            preds_with_objects[:, 0:4], "cxcywh", "xyxy"
        )

        # choose predictions from argmaxed IoU along label dim to get best prediction per label
        prediction_matrix = ops.box_iou(
            labels_with_objects[:, 1:5], preds_with_objects[:, 0:4]
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
        final_preds = reformatted_preds[labels_mask]
        final_preds[:, 0:4] = ops.box_convert(final_preds[:, 0:4], "cxcywh", "xyxy")

    return final_preds, labels_with_objects
