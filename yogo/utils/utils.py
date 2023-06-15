#! /usr/bin/env python3

import time
import wandb
import torch

from contextlib import contextmanager

import PIL
import torchvision.ops as ops
import torchvision.transforms as transforms

from typing import (
    Optional,
    Sequence,
    Generator,
    TypeVar,
    Union,
    List,
    Literal,
    Tuple,
    get_args,
)


T = TypeVar("T")
BoxFormat = Literal["xyxy", "cxcywh"]


@contextmanager
def Timer(description: str, precision: int = 5, post_print: bool = False):
    """Context manager for timing code execution.

    Args:
        description (str): description of code to be timed
        precision (float): number of digits to print after decimal point
        post_print (bool): whether to print information only after leaving the context
    """
    try:
        start_time = time.perf_counter()
        if not post_print:
            print(f"{description}...", end=" ", flush=True)
        yield
    finally:
        end_time = time.perf_counter()
        print(
            f"{str(description) + ' ' if post_print else ''}{end_time - start_time:.{precision}f} s"
        )


def get_wandb_roc(
    fpr: Union[Sequence, Sequence[Sequence]],
    tpr: Sequence[Sequence],
    thresholds: Union[Sequence, Sequence[Sequence]],
    classes: Sequence,
):
    if not isinstance(fpr, Sequence):
        raise TypeError(f"Expected fpr to be an array instead got {type(fpr)}")

    if not isinstance(tpr, Sequence):
        raise TypeError(f"Expected tpr to be an array instead got {type(tpr)}")

    for y in tpr:
        if not isinstance(y, Sequence):
            raise TypeError(
                f"Expected tpr to be an array of arrays instead got {type(y)}"
            )

    if not isinstance(fpr[0], Sequence) or isinstance(fpr[0], (str, bytes)):
        fpr = [fpr for _ in range(len(tpr))]

    if not isinstance(thresholds[0], Sequence) or isinstance(
        thresholds[0], (str, bytes)
    ):
        thresholds = [thresholds for _ in range(len(tpr))]

    assert len(fpr) == len(tpr), "Number of fprs and tprs must match"
    assert len(classes) == len(tpr), "Number of classes and tprs must match"

    data = [
        [x, y, thr, classes[i]]
        for i, (xx, yy, thrs) in enumerate(zip(fpr, tpr, thresholds))
        for x, y, thr in zip(xx, yy, thrs)
    ]

    return wandb.Table(data=data, columns=["fpr", "tpr", "threshold", "class"])


def get_wandb_confusion(
    confusion_data: torch.Tensor,
    class_names: List[str],
    title: str = "confusion matrix",
):
    nc1, nc2 = confusion_data.shape
    assert (
        nc1 == nc2 == len(class_names)
    ), f"nc1 != nc2 != len(class_names)! (nc1 = {nc1}, nc2 = {nc2}, class_names = {class_names})"

    L = []
    for i in range(nc1):
        for j in range(nc2):
            # annoyingly, wandb will sort the matrix by row/col names. sad!
            # fix the order we want by prepending the index of the class.
            L.append(
                (
                    f"{i} - {class_names[i]}",
                    f"{j} - {class_names[j]}",
                    confusion_data[i, j],
                )
            )

    return wandb.plot_table(
        "wandb/confusion_matrix/v1",
        wandb.Table(
            columns=["Actual", "Predicted", "nPredictions"],
            data=L,
        ),
        {
            "Actual": "Actual",
            "Predicted": "Predicted",
            "nPredictions": "nPredictions",
        },
        {"title": title},
    )


def iter_in_chunks(s: Sequence[T], n: int = 1) -> Generator[Sequence[T], None, None]:
    for i in range(0, len(s), n):
        yield s[i : i + n]


def format_preds(
    batch_pred: torch.Tensor,
    thresh: float = 0.5,
    iou_thresh: float = 0.5,
    box_format: BoxFormat = "cxcywh",
) -> torch.Tensor:
    """
    formats batch_pred, from YOGO, into [N,pred_shape], after applying NMS and
    thresholding objectness at thresh.

    box_format specifies the returned box format
    """
    if len(batch_pred.shape) != 3:
        raise ValueError(
            "argument to format_pred should be unbatched result - "
            f"shape should be (pred_shape, Sy, Sx), got {batch_pred.shape}"
        )
    elif box_format not in get_args(BoxFormat):
        raise ValueError(
            f"invalid box format {box_format}; valid box formats are {get_args(BoxFormat)}"
        )

    pred_shape, Sy, Sx = batch_pred.shape

    reformatted_preds = batch_pred.view(pred_shape, Sx * Sy).T

    # Filter for objectness first
    objectness_mask = (reformatted_preds[:, 4] > thresh).bool()
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
            preds[:, 4],
            iou_threshold=iou_thresh,
        )
    else:
        keep_idxs = torch.arange(len(preds))

    return preds[keep_idxs]


def _format_tensor_for_rects(
    rects: torch.Tensor,
    img_h: int,
    img_w: int,
    thresh: float = 0.5,
    iou_thresh: float = 0.5,
) -> torch.Tensor:
    pred_dim, Sy, Sx = rects.shape

    formatted_preds = format_preds(
        rects,
        thresh=thresh,
        iou_thresh=iou_thresh,
        box_format="xyxy",
    )

    N = formatted_preds.shape[0]
    formatted_rects = torch.zeros((N, 6), device=formatted_preds.device)
    formatted_rects[:, (0, 2)] = img_w * formatted_preds[:, (0, 2)]
    formatted_rects[:, (1, 3)] = img_h * formatted_preds[:, (1, 3)]
    formatted_rects[:, 4] = torch.argmax(formatted_preds[:, 5:], dim=1)
    formatted_rects[:, 5] = formatted_preds[:, 4]
    return formatted_rects


def bbox_colour(label: str, opacity: float = 1.0) -> Tuple[int, int, int, int]:
    if not (0 <= opacity <= 1):
        raise ValueError(f"opacity must be between 0 and 1, got {opacity}")
    if label in ("healthy", "0"):
        return (0, 255, 0, int(opacity * 255))
    elif label in ("misc", "6"):
        return (0, 0, 0, int(opacity * 255))
    return (255, 0, 0, int(opacity * 255))


def draw_yogo_prediction(
    img: torch.Tensor,
    prediction: torch.Tensor,
    thresh: float = 0.5,
    iou_thresh: float = 0.5,
    labels: Optional[List[str]] = None,
    images_are_normalized: bool = False,
) -> PIL.Image.Image:
    """Given an image and a prediction, return a PIL Image with bounding boxes

    args:
        img: 2d torch.Tensor of shape (h, w) or (1, h, w). We will `torch.uint8` your tensor!
        prediction: torch.tensor of shape (pred_dim, Sy, Sx) or (1, pred_dim, Sy, Sx)
        thresh: objectness threshold
        iou_thresh: IoU threshold for non-maximal supression (i.e. removal of doubled bboxes)
        labels: list of label names for displaying
    """
    img, prediction = img.squeeze(), prediction.squeeze()

    # NOTE I dont know how I feel about this - perhaps it is better to accept
    # only u8s in range [0,255]
    if images_are_normalized:
        img *= 255

    img = img.to(torch.uint8)

    if img.ndim != 2:
        raise ValueError(
            "img must be 2-dimensional (i.e. grayscale), "
            f"but has {img.ndim} dimensions"
        )
    elif prediction.ndim != 3:
        raise ValueError(
            "prediction must be 'unbatched' (i.e. shape (pred_dim, Sy, Sx) or "
            f"(1, pred_dim, Sy, Sx)) - got shape {prediction.shape} "
        )

    img_h, img_w = img.shape

    formatted_rects: Union[torch.Tensor, List] = _format_tensor_for_rects(
        prediction,
        img_h=img_h,
        img_w=img_w,
        thresh=thresh,
        iou_thresh=iou_thresh,
    )

    pil_img = transforms.ToPILImage()(img[None, ...])

    rgb = PIL.Image.new("RGBA", pil_img.size)
    rgb.paste(pil_img)
    draw = PIL.ImageDraw.Draw(rgb)  # type: ignore

    for r in formatted_rects:
        r = list(r)
        label = labels[int(r[4])] if labels is not None else str(r[4])
        draw.rectangle(r[:4], outline=bbox_colour(label))
        draw.text((r[0], r[1]), label, (0, 0, 0, 255))

    return rgb
