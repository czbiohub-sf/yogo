#! /usr/bin/env python3

import json
import torch
import signal
import datetime
import warnings

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Optional, Literal

from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop

from yogo.model import YOGO
from yogo.utils.argparsers import infer_parser
from yogo.data.image_path_dataset import ZarrDataset, get_dataset, collate_fn
from yogo.data.yogo_dataloader import choose_dataloader_num_workers
from yogo.utils import (
    draw_yogo_prediction,
    format_preds,
    choose_device,
    format_to_numpy,
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
):
    for fname, pred_slice in zip(fnames, batch_preds):
        preds = format_preds(
            pred_slice,
            obj_thresh=obj_thresh,
            iou_thresh=iou_thresh,
        )

        pred_string = "\n".join(
            f"{argmax(pred[5:])} {pred[0]} {pred[1]} {pred[2]} {pred[3]}"
            for pred in preds
        )
        with open(fname, "w") as f:
            f.write(pred_string)


def get_prediction_class_counts(
    batch_preds: torch.Tensor,
    obj_thresh=0.5,
    iou_thresh=0.5,
    min_class_confidence_threshold: float = 0,
) -> torch.Tensor:
    """
    Count the number of predictions of each class, by argmaxing the class predictions
    """
    bs, pred_dim, Sy, Sx = batch_preds.shape
    num_classes = pred_dim - 5
    tot_class_sum = torch.zeros(num_classes, dtype=torch.long)

    for pred_slice in batch_preds:
        preds = format_preds(
            pred_slice,
            obj_thresh=obj_thresh,
            iou_thresh=iou_thresh,
            min_class_confidence_threshold=min_class_confidence_threshold,
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


def get_model_name_from_pth(path_to_pth: Union[str, Path]) -> Optional[str]:
    return torch.load(Path(path_to_pth), map_location="cpu").get("model_name", None)


def write_metadata(metadata_path: Path, **kwargs):
    """
    very simply writes a json file with the kwargs
    """
    with open(metadata_path.with_suffix(".json"), "w") as f:
        json.dump(kwargs, f, indent=4)


@torch.no_grad()
def predict(
    path_to_pth: str,
    *,
    path_to_images: Optional[Path] = None,
    path_to_zarr: Optional[Path] = None,
    output_dir: Optional[str] = None,
    draw_boxes: bool = False,
    save_preds: bool = False,
    save_npy: bool = False,
    class_names: Optional[List[str]] = None,
    count_predictions: bool = False,
    batch_size: int = 64,
    obj_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    vertical_crop_height: Optional[int] = None,
    use_tqdm: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    output_img_ftype: Literal[".png", ".tif", ".tiff"] = ".png",
    requested_num_workers: Optional[int] = None,
    min_class_confidence_threshold: float = 0.0,
    half: bool = False,
    return_full_predictions: bool = False,
) -> Optional[torch.Tensor]:
    """
    This is a bit of a gargantuan function. It handles `yogo infer` as well as
    general inference using YOGO. It can be used directly, but most of the time
    is invoked through the CLI.

    Mostly, see `yogo infer --help` for the help. Here is a recapitulation (plus
    some extras):

        path_to_pth: path to .pth file defining the model
        path_to_images: path to image or images; if path_to_images is not None, path_to_zarr must be None
        path_to_zarr: path to zarr file; if path_to_zarr is not None, path_to_images must be None
        output_dir: directory to save predictions or draw-boxes in YOGO format
        output_img_ftype: output image filetype for bounding boxes
        draw_boxes: whether to draw boxes in YOGO format
        save_preds: whether to save predictions in YOGO format
        save_npy: whether to save predictions in .npy format
        class_names: list of class names
        count_predictions: whether to count the number of predictions
        batch_size: batch size
        obj_thresh: object threshold
        iou_thresh: iou threshold
        vertical_crop_height: vertical crop height
        use_tqdm: whether to use tqdm
        device: device to run infer on
        requested_num_workers: number of workers to use
        min_class_confidence_threshold: minimum confidence threshold for class
        half: whether to use half precision
        return_full_predictions: whether to return full predictions; useful for getting YOGO predictions
                                 from python
    """
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

    device = torch.device(device or choose_device())

    model, cfg = YOGO.from_pth(Path(path_to_pth), inference=True)
    model.eval()
    model.to(device)

    transforms: List[torch.nn.Module] = []

    img_h, img_w = model.get_img_size()
    if vertical_crop_height:
        vertical_crop_height_px = (vertical_crop_height * img_h).round()
        crop = CenterCrop((int(vertical_crop_height_px.item()), int(img_w.item())))
        transforms.append(crop)
        model.resize_model(int(vertical_crop_height_px.item()))
        img_h = vertical_crop_height_px

    # these three lines are correctly typed; dunno how to convince mypy
    assert model.img_size.numel() == 2, f"YOGO model must be 2D, is {model.img_size}"  # type: ignore
    img_in_h = int(model.img_size[0].item())  # type: ignore
    img_in_w = int(model.img_size[1].item())  # type: ignore

    dummy_input = torch.randint(0, 256, (1, 1, img_in_h, img_in_w), device=device)

    if device.type == "cuda":
        # TODO expand accepted device types!
        model_jit = torch.compile(model)
    else:
        model_jit = model

    output_shape = model_jit(dummy_input).shape
    num_classes = output_shape[1] - 5

    if class_names is not None:
        if len(class_names) != num_classes:
            raise ValueError(
                f"expected {num_classes} class names, got {len(class_names)}"
            )

    image_dataset = get_dataset(
        path_to_images=path_to_images,
        path_to_zarr=path_to_zarr,
        image_transforms=transforms,
        normalize_images=bool(model.normalize_images),
    )

    if isinstance(image_dataset, ZarrDataset):
        warnings.warn(
            "There is some bug with multiprocessed reading "
            "of a zarr array that hasn't yet been squashed. "
            "The number of dataloader workers must be fixed "
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

    # this tensor can be really big, so only create it if we need it
    if return_full_predictions:
        results = torch.zeros(
            (len(image_dataset), output_shape[1], output_shape[2], output_shape[3]),
        )

    if save_npy:
        np_results = []

    if count_predictions:
        tot_counts = torch.zeros((num_classes,))

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

        # gross! device-type is checked even if enabled=False, which means we
        # have to just tell autocast that device type is always cuda.
        with torch.cuda.amp.autocast(
            enabled=half and device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            res = model_jit(img_batch.to(device))

        if draw_boxes:
            for img_idx in range(img_batch.shape[0]):
                bbox_img = draw_yogo_prediction(
                    img=img_batch[img_idx, ...],
                    prediction=res[img_idx, ...],
                    obj_thresh=obj_thresh,
                    iou_thresh=iou_thresh,
                    min_class_confidence_threshold=min_class_confidence_threshold,
                    labels=class_names,
                    images_are_normalized=model.normalize_images,
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
        if save_preds:
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
            )
        if save_npy:
            res = res.cpu().numpy()

            for j in range(res.shape[0]):
                img_index = (i * batch_size) + j
                parsed = format_to_numpy(
                    img_index,
                    res[j, ...],
                    int(img_h.item()),
                    int(img_w.item()),
                )
                np_results.append(parsed)

        if count_predictions:
            tot_counts += get_prediction_class_counts(
                res.cpu(),
                obj_thresh=obj_thresh,
                iou_thresh=iou_thresh,
                min_class_confidence_threshold=min_class_confidence_threshold,
            )

        # sometimes we return a number of images less than the batch size,
        # namely when len(image_dataset) % batch_size != 0
        if return_full_predictions:
            results[i * batch_size : i * batch_size + res.shape[0], ...] = res

        pbar.update(res.shape[0])

    pbar.close()

    if count_predictions:
        print(list(zip(class_names or range(num_classes), map(int, tot_counts))))

    # Save the numpy array
    if save_npy:
        pred_tensors = np.hstack(np_results)

        if path_to_images:
            filename = Path(path_to_images).resolve().parent.stem
        elif path_to_zarr:
            filename = Path(path_to_zarr).resolve().stem

        if output_dir is not None:
            fp = Path(output_dir).resolve() / Path(filename).with_suffix(".npy")
        else:
            fp = Path.cwd().resolve() / Path(filename).with_suffix(".npy")

        np.save(fp, pred_tensors)

        write_metadata(
            fp.with_suffix(".json"),
            run_name=fp.with_suffix("").name,
            model_name=get_model_name_from_pth(path_to_pth),
            obj_thresh=obj_thresh,
            iou_thresh=iou_thresh,
            vertical_crop_height_px=img_h.item(),
            write_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    if return_full_predictions:
        return results

    return None


def do_infer(args):
    predict(
        args.pth_path,
        path_to_images=args.path_to_images,
        path_to_zarr=args.path_to_zarr,
        output_dir=args.output_dir,
        draw_boxes=args.draw_boxes,
        save_preds=args.save_preds,
        save_npy=args.save_npy,
        class_names=args.class_names,
        obj_thresh=args.obj_thresh,
        iou_thresh=args.iou_thresh,
        batch_size=args.batch_size,
        device=args.device,
        use_tqdm=args.use_tqdm,
        vertical_crop_height=args.crop_height,
        count_predictions=args.count,
        output_img_ftype=args.output_img_filetype,
        min_class_confidence_threshold=args.min_class_confidence_threshold,
        half=args.half,
    )


if __name__ == "__main__":
    parser = infer_parser()
    args = parser.parse_args()
    do_infer(args)
