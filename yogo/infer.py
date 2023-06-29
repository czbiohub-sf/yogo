#! /usr/bin/env python3

import zarr
import math
import torch
import signal
import warnings

import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from pathlib import Path
from collections.abc import Sized
from typing import List, Union, Optional, Callable, Tuple, cast, Literal

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, CenterCrop

from yogo.model import YOGO
from yogo.utils.argparsers import infer_parser
from yogo.utils import draw_yogo_prediction, format_preds
from yogo.data.dataset import read_grayscale, YOGO_CLASS_ORDERING


# lets us ctrl-c to exit while matplotlib is showing stuff
signal.signal(signal.SIGINT, signal.SIG_DFL)


class ImageAndIdDataset(Dataset, Sized):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        raise NotImplementedError


class ImagePathDataset(ImageAndIdDataset):
    def __init__(
        self,
        root: Union[str, Path],
        image_transforms: List[nn.Module] = [],
        loader: Callable[
            [
                Union[str, Path],
            ],
            torch.Tensor,
        ] = read_grayscale,
        normalize_images: bool = False,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"{self.root} does not exist")

        self.image_paths = self.make_dataset(self.root)

        self.transform = Compose(image_transforms)
        self.loader = loader
        self.normalize_images = normalize_images

    def make_dataset(self, path_to_data: Path) -> np.ndarray:
        if path_to_data.is_file() and path_to_data.suffix == ".png":
            img_paths = [path_to_data]
        else:
            img_paths = [
                p for p in path_to_data.glob("*.png") if not p.name.startswith(".")
            ]
        if len(img_paths) == 0:
            raise FileNotFoundError(f"{str(path_to_data)} does not contain any images")
        return np.array(img_paths).astype(np.string_)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = str(self.image_paths[idx], encoding="utf-8")
        image = self.loader(image_path).to(torch.float16)
        image = self.transform(image)
        if self.normalize_images:
            image = image / 255
        return image, image_path


class ZarrDataset(ImageAndIdDataset):
    def __init__(
        self,
        zarr_path: Union[str, Path],
        image_name_from_idx: Optional[Callable[[int], str]] = None,
        image_transforms: List[nn.Module] = [],
        normalize_images: bool = False,
    ):
        """Note

        zip files can be corrupted easily, so be aware that this
        may run into some zarr corruption issues. ImagePathDataset
        is pretty failsafe
        """
        self.zarr_path = Path(zarr_path)
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"{self.zarr_path} does not exist")

        self.zarr_store = zarr.open(str(self.zarr_path), mode="r")

        self.image_name_from_idx = image_name_from_idx or self._image_name_from_idx

        self.transform = Compose(image_transforms)
        self.normalize_images = normalize_images
        self._N = int(math.log(len(self), 10) + 1)

    def _image_name_from_idx(self, idx: int) -> str:
        return f"img_{idx:0{self._N}}.png"

    def __len__(self) -> int:
        return (
            self.zarr_store.initialized
            if isinstance(self.zarr_store, zarr.Array)
            else len(self.zarr_store)
        )

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        image = (
            self.zarr_store[:, :, idx]
            if isinstance(self.zarr_store, zarr.Array)
            else self.zarr_store[idx][:]
        )[None, ...]
        image = torch.from_numpy(image).to(torch.float16)
        image = self.transform(image)
        if self.normalize_images:
            image = image / 255

        return image, self.image_name_from_idx(idx)


def collate_fn(
    batch: List[Tuple[torch.Tensor, str]]
) -> Tuple[torch.Tensor, Tuple[str]]:
    images, fnames = zip(*batch)
    return torch.stack(images), cast(Tuple[str], fnames)


def get_dataset(
    path_to_images: Optional[Path] = None,
    path_to_zarr: Optional[Path] = None,
    image_transforms: List[nn.Module] = [],
    normalize_images: bool = False,
) -> ImageAndIdDataset:
    if path_to_images is not None and path_to_zarr is not None:
        raise ValueError(
            "can only take one of 'path_to_images' or 'path_to_zarr', but got both"
        )
    elif path_to_images is not None:
        return ImagePathDataset(
            path_to_images,
            image_transforms=image_transforms,
            normalize_images=normalize_images,
        )
    elif path_to_zarr is not None:
        return ZarrDataset(
            path_to_zarr,
            image_transforms=image_transforms,
            normalize_images=normalize_images,
        )
    else:
        raise ValueError("one of 'path_to_images' or 'path_to_zarr' must not be None")


def argmax(arr):
    return max(range(len(arr)), key=arr.__getitem__)


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_dataloader_num_workers(dataset_size):
    if dataset_size < 1000:
        return 0
    return min(torch.multiprocessing.cpu_count(), 32)


def save_predictions(
    fnames,
    batch_preds,
    obj_thresh=0.5,
    iou_thresh=0.5,
    aspect_thresh: Optional[float] = None,
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
            aspect_thresh=aspect_thresh,
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
    aspect_thresh: Optional[float] = None,
) -> torch.Tensor:
    """
    Count the number of predictions of each class, by argmaxing the class predictions
    """
    tot_class_sum = torch.zeros(len(YOGO_CLASS_ORDERING), dtype=torch.long)
    for pred_slice in batch_preds:
        preds = format_preds(
            pred_slice,
            obj_thresh=obj_thresh,
            iou_thresh=iou_thresh,
            aspect_thresh=aspect_thresh,
        )
        if preds.numel() == 0:
            continue  # ignore no predictions
        classes = preds[:, 5:]
        tot_class_sum += count_cells_for_formatted_preds(classes)
    return tot_class_sum


def count_cells_for_formatted_preds(
    formatted_class_predictions: torch.Tensor,
) -> torch.Tensor:
    """
    Count the number of predictions in each class of the prediction tensor
    Expecting shape of (N, num_classes), and each row must sum to 1
    """
    if not len(formatted_class_predictions.shape) == 2:
        raise ValueError(
            "expected formatted_class_predictions to be shape (N, num_classes); "
            "got {formatted_class_predictions.shape}"
        )
    n_predictions, n_classes = formatted_class_predictions.shape
    class_predictions = formatted_class_predictions.argmax(dim=1)
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
    count_predictions: bool = False,
    batch_size: int = 64,
    obj_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    aspect_thresh: Optional[float] = None,
    label: Optional[str] = None,
    vertical_crop_height_px: Optional[int] = None,
    use_tqdm: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    output_img_ftype: Literal[".png", ".tif", ".tiff"] = ".png",
) -> Optional[torch.Tensor]:
    if save_preds and draw_boxes:
        raise ValueError(
            "cannot save predictions in YOGO format and draw_boxes at the same time"
        )
    elif output_dir is not None and not (save_preds or draw_boxes):
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
    model.to(device)
    model.eval()

    img_h, img_w = model.get_img_size()

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

    image_dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn,
        num_workers=choose_dataloader_num_workers(len(image_dataset)),
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

    for i, (img_batch, fnames) in enumerate(image_dataloader):
        res = model(img_batch.to(device)).to("cpu")

        assert torch.all(res <= 1), f"returned tensor w/ max value {res.max()}"

        if draw_boxes:
            for img_idx in range(img_batch.shape[0]):
                bbox_img = draw_yogo_prediction(
                    img=img_batch[img_idx, ...],
                    prediction=res[img_idx, ...],
                    iou_thresh=0.5,
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
        elif save_preds:
            assert (
                output_dir is not None
            ), "output_dir must not be None if save_preds is True"
            out_fnames = [
                Path(output_dir) / Path(fname).with_suffix(".txt").name
                for fname in fnames
            ]
            save_predictions(out_fnames, res, obj_thresh=0.5, label=label)
        else:
            # sometimes we return a number of images less than the batch size,
            # namely when len(image_dataset) % batch_size != 0
            results[i * batch_size : i * batch_size + res.shape[0], ...] = res.cpu()

        pbar.update(res.shape[0])

    pbar.close()

    if count_predictions:
        counts = get_prediction_class_counts(results).tolist()
        tot_cells = sum(counts)
        print(
            list(
                zip(
                    YOGO_CLASS_ORDERING,
                    counts,
                    [round(c / tot_cells, 4) for c in counts],
                )
            )
        )

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
        draw_boxes=args.draw_boxes,
        obj_thresh=args.obj_thresh,
        iou_thresh=args.iou_thresh,
        aspect_thresh=args.aspect_thresh,
        batch_size=args.batch_size,
        use_tqdm=(args.output_dir is not None or args.draw_boxes or args.count),
        vertical_crop_height_px=(
            round(772 * args.crop_height) if args.crop_height is not None else None
        ),
        count_predictions=args.count,
        output_img_ftype=args.output_img_filetype,
    )


if __name__ == "__main__":
    parser = infer_parser()
    args = parser.parse_args()
    do_infer(args)
