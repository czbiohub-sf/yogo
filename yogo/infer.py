#! /usr/bin/env python3

import torch
import signal
import warnings

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Optional, Literal

from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop

from yogo.model import YOGO
from yogo.data import YOGO_CLASS_ORDERING
from yogo.utils.argparsers import infer_parser
from yogo.data.image_path_dataset import ZarrDataset, get_dataset, collate_fn
from yogo.data.yogo_dataloader import choose_dataloader_num_workers
from yogo.utils import (
    draw_yogo_prediction,
    format_preds,
    choose_device,
    parse_prediction_tensor,
)


# lets us ctrl-c to exit while matplotlib is showing stuff
signal.signal(signal.SIGINT, signal.SIG_DFL)


def argmax(arr):
    return max(range(len(arr)), key=arr.__getitem__)


def save_predictions(
    fnames,
    batch_preds,
    obj_thresh=0.5,
    iou_thresh=0.5,
    label: Optional[str] = None,
):
    bs, pred_shape, Sy, Sx = batch_preds.shape

    if label is not None:
        label_idx = YOGO_CLASS_ORDERING.index(label)
    else:
        # var is not used
        label_idx = None

    for fname, pred_slice in zip(fnames, batch_preds):
        preds = format_preds(
            pred_slice,
            obj_thresh=obj_thresh,
            iou_thresh=iou_thresh,
        )

        pred_string = "\n".join(
            f"{argmax(pred[5:]) if label is None else label_idx} {pred[0]} {pred[1]} {pred[2]} {pred[3]}"
            for pred in preds
        )
        with open(fname, "w") as f:
            f.write(pred_string)


def get_prediction_class_counts(
    batch_preds: torch.Tensor,
    obj_thresh=0.5,
    iou_thresh=0.5,
    min_class_confidence_threshold: float = 0,
    heatmap_mask_path: Optional[Path] = None,
) -> torch.Tensor:
    """
    Count the number of predictions of each class, by argmaxing the class predictions
    """
    tot_class_sum = torch.zeros(len(YOGO_CLASS_ORDERING), dtype=torch.long)
    heatmap_mask = None if heatmap_mask_path is None else np.load(heatmap_mask_path)
    for pred_slice in batch_preds:
        preds = format_preds(
            pred_slice,
            obj_thresh=obj_thresh,
            iou_thresh=iou_thresh,
            min_class_confidence_threshold=min_class_confidence_threshold,
            heatmap_mask=heatmap_mask,
        )
        if preds.numel() == 0:
            continue  # ignore no predictions
        classes = preds[:, 5:]
        tot_class_sum += count_cells_for_formatted_preds(classes)
    return tot_class_sum


def count_cells_for_formatted_preds(
    formatted_class_predictions: torch.Tensor,
    min_confidence_threshold: Optional[float] = None,
) -> torch.Tensor:
    """
    Count the number of predictions in each class of the prediction tensor
    Expecting shape of (N, num_classes), and each row must sum to 1.

    if min_confidence_threshold is not None, this will ignore predictions
    with a maximum confidence below min_confidence_threshold. Should be between
    0 and 1.
    """
    if not len(formatted_class_predictions.shape) == 2:
        raise ValueError(
            "expected formatted_class_predictions to be shape (N, num_classes); "
            "got {formatted_class_predictions.shape}"
        )
    if min_confidence_threshold is not None:
        if min_confidence_threshold < 0 or min_confidence_threshold > 1:
            raise ValueError(
                "min_confidence_threshold should be between 0 and 1; "
                f"is {min_confidence_threshold}"
            )
    else:
        min_confidence_threshold = 0

    _, n_classes = formatted_class_predictions.shape

    values, indices = formatted_class_predictions.max(dim=1)
    mask = values > min_confidence_threshold
    class_predictions = indices[mask]

    return torch.nn.functional.one_hot(class_predictions, num_classes=n_classes).sum(
        dim=0
    )


@torch.no_grad()
def predict(
    path_to_pth: str,
    path_to_images: Optional[Path] = None,
    path_to_zarr: Optional[Path] = None,
    output_dir: Optional[str] = None,
    save_preds: bool = False,
    draw_boxes: bool = False,
    save_npy: bool = False,
    count_predictions: bool = False,
    batch_size: int = 64,
    obj_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    label: Optional[str] = None,
    vertical_crop_height_px: Optional[int] = None,
    use_tqdm: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    output_img_ftype: Literal[".png", ".tif", ".tiff"] = ".png",
    requested_num_workers: Optional[int] = None,
    min_class_confidence_threshold: float = 0.0,
    heatmap_mask_path: Optional[Path] = None,
) -> Optional[torch.Tensor]:
    if save_preds and draw_boxes:
        raise ValueError(
            "cannot save predictions in YOGO format and draw_boxes at the same time"
        )
    elif output_dir is not None and not (save_preds or draw_boxes or save_npy):
        warnings.warn(
            f"output dir is not None (is {output_dir}), but it will not be used "
            "since save_preds and draw_boxes are both false"
        )
    elif output_dir is not None:
        Path(output_dir).mkdir(exist_ok=True, parents=False)
    elif save_preds:
        raise ValueError("output_dir must not be None if save_preds is True")
    elif output_img_ftype not in [".png", ".tif", ".tiff"]:
        raise ValueError(
            "only .png, .tif, and .tiff are supported for output img "
            "filetype; got {output_img_ftype}"
        )

    device = device or choose_device()

    model, cfg = YOGO.from_pth(Path(path_to_pth), inference=True)
    model.eval()
    model.to(device)
    # mypy thinks that compile isn't in the torch package, but it is
    model_jit = torch.compile(model)  # type: ignore

    transforms: List[torch.nn.Module] = []

    if vertical_crop_height_px:
        crop = CenterCrop((vertical_crop_height_px, 1032))
        transforms.append(crop)
        model.resize_model(vertical_crop_height_px)

    image_dataset = get_dataset(
        path_to_images=path_to_images,
        path_to_zarr=path_to_zarr,
        image_transforms=transforms,
        normalize_images=cfg["normalize_images"],
    )

    if isinstance(image_dataset, ZarrDataset):
        warnings.warn(
            "there is some bug with multiprocessed reading "
            "of a zarr array that I haven't figured out yet. "
            "We have to fix the number of dataloader workers "
            "to 0, so this will probably be slow. It will be "
            "much faster to use --path-to-images instead."
        )
        num_workers = 0
    else:
        num_workers = choose_dataloader_num_workers(
            len(image_dataset), requested_num_workers=requested_num_workers
        )

    image_dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    pbar = tqdm(
        disable=not use_tqdm,
        unit="images",
        total=len(image_dataset),
    )

    Sx, Sy = model.get_grid_size()

    # this tensor can be really big, so only create it if we need it
    if not (draw_boxes or save_preds):
        results = torch.zeros(
            (len(image_dataset), len(YOGO_CLASS_ORDERING) + 5, Sy, Sx)
        )
    if save_npy:
        np_results = []
        img_h: int = (
            vertical_crop_height_px if vertical_crop_height_px is not None else 772
        )

    file_iterator = enumerate(image_dataloader)
    while True:
        # attempting to be forgiving to malformed images, which sometimes occurs
        # when exporting zip files
        try:
            i, (img_batch, fnames) = next(file_iterator)
        except StopIteration:
            break
        except RuntimeError as e:
            warnings.warn(f"got error {e}; continuing")
            continue

        res = model_jit(img_batch.to(device))

        if draw_boxes:
            for img_idx in range(img_batch.shape[0]):
                bbox_img = draw_yogo_prediction(
                    img=img_batch[img_idx, ...],
                    prediction=res[img_idx, ...],
                    obj_thresh=obj_thresh,
                    iou_thresh=iou_thresh,
                    labels=YOGO_CLASS_ORDERING,
                    images_are_normalized=cfg["normalize_images"],
                )
                if output_dir is not None:
                    out_fname = (
                        Path(output_dir)
                        / Path(fnames[img_idx]).with_suffix(output_img_ftype).name
                    )
                    # don't need to compress these, we delete later
                    # mypy thinks that you can't save a PIL Image which is false
                    bbox_img.save(out_fname, compress_level=1)  # type: ignore
                else:
                    fig, ax = plt.subplots()
                    ax.set_axis_off()
                    ax.imshow(bbox_img)
                    plt.show()

                plt.clf()
                plt.close()

        elif save_preds:
            assert (
                output_dir is not None
            ), "output_dir must not be None if save_preds is True"
            out_fnames = [
                Path(output_dir) / Path(fname).with_suffix(".txt").name
                for fname in fnames
            ]
            save_predictions(
                out_fnames,
                res,
                obj_thresh=obj_thresh,
                iou_thresh=iou_thresh,
                label=label,
            )
        elif save_npy:
            for j in range(res.shape[0]):
                yogo_res = res[j, ...].clone().cpu().numpy()

                index = (i * batch_size) + j
                parsed = parse_prediction_tensor(index, yogo_res, img_h, 1032)
                np_results.append(parsed)
        else:
            # sometimes we return a number of images less than the batch size,
            # namely when len(image_dataset) % batch_size != 0
            results[i * batch_size : i * batch_size + res.shape[0], ...] = res

        pbar.update(res.shape[0])

    pbar.close()

    if count_predictions:
        counts = get_prediction_class_counts(
            results,
            obj_thresh=obj_thresh,
            iou_thresh=iou_thresh,
            min_class_confidence_threshold=min_class_confidence_threshold,
            heatmap_mask_path=heatmap_mask_path,
        ).tolist()
        tot_cells = sum(counts)
        print(
            list(
                zip(
                    YOGO_CLASS_ORDERING,
                    counts,
                    [0 if tot_cells == 0 else round(c / tot_cells, 4) for c in counts],
                )
            )
        )

    # Save the numpy array
    if save_npy:
        pred_tensors = np.hstack(np_results)

        if path_to_images:
            filename = Path(path_to_images).stem
        elif path_to_zarr:
            filename = Path(path_to_zarr).stem

        if output_dir is not None:
            fp = Path(output_dir) / Path(filename).with_suffix(".npy")
            np.save(f"{fp}", pred_tensors)
        else:
            # Save it to the current location
            np.save(f"{filename}.npy", pred_tensors)

    if not (draw_boxes or save_preds):
        return results

    return None


def do_infer(args):
    predict(
        args.pth_path,
        path_to_images=args.path_to_images,
        path_to_zarr=args.path_to_zarr,
        output_dir=args.output_dir,
        save_preds=args.save_preds,
        save_npy=args.save_npy,
        draw_boxes=args.draw_boxes,
        obj_thresh=args.obj_thresh,
        iou_thresh=args.iou_thresh,
        batch_size=args.batch_size,
        device=args.device,
        use_tqdm=(args.output_dir is not None or args.draw_boxes or args.count),
        vertical_crop_height_px=(
            round(772 * args.crop_height) if args.crop_height is not None else None
        ),
        count_predictions=args.count,
        output_img_ftype=args.output_img_filetype,
        min_class_confidence_threshold=args.min_class_confidence_threshold,
        heatmap_mask_path=args.heatmap_mask_path,
    )


if __name__ == "__main__":
    parser = infer_parser()
    args = parser.parse_args()
    do_infer(args)
