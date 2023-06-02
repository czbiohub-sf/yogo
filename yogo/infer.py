#! /usr/bin/env python3

import zarr
import math
import torch
import signal

import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from pathlib import Path
from collections.abc import Sized
from typing import List, Union, Optional, Callable, Tuple, cast

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from yogo.model import YOGO
from yogo.utils.argparsers import infer_parser
from yogo.utils import draw_yogo_prediction, format_preds
from yogo.data.dataset import read_grayscale, YOGO_CLASS_ORDERING


# lets us ctrl-c to exit while matplotlib is showing stuff
signal.signal(signal.SIGINT, signal.SIG_DFL)


def argmax(arr):
    return max(range(len(arr)), key=arr.__getitem__)


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_preds(fnames, batch_preds, thresh=0.5, label: Optional[str] = None):
    bs, pred_shape, Sy, Sx = batch_preds.shape

    if label is not None:
        label_idx = YOGO_CLASS_ORDERING.index(label)
    else:
        # var is not used
        label_idx = None

    for fname, batch_pred in zip(fnames, batch_preds):
        preds = format_preds(batch_pred)

        pred_string = "\n".join(
            f"{argmax(pred[5:]) if label is None else label_idx} {pred[0]} {pred[1]} {pred[2]} {pred[3]}"
            for pred in preds
        )
        with open(fname, "w") as f:
            f.write(pred_string)


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
        )
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


@torch.no_grad()
def predict(
    path_to_pth: str,
    path_to_images: Optional[Path] = None,
    path_to_zarr: Optional[Path] = None,
    output_dir: Optional[str] = None,
    thresh: float = 0.5,
    draw_boxes: bool = False,
    batch_size: int = 16,
    use_tqdm: bool = False,
    device: Union[str, torch.device] = "cpu",
    print_results: bool = False,
    label: Optional[str] = None,
) -> Optional[torch.Tensor]:
    model, cfg = YOGO.from_pth(Path(path_to_pth), inference=True)
    model.to(device)
    model.eval()

    img_h, img_w = model.get_img_size()
    Sx, Sy = model.get_grid_size()

    image_dataset = get_dataset(
        path_to_images=path_to_images,
        path_to_zarr=path_to_zarr,
        normalize_images=cfg["normalize_images"],
    )

    image_dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn,
        num_workers=min(torch.multiprocessing.cpu_count(), 32),
    )

    if output_dir is not None:
        Path(output_dir).mkdir(exist_ok=True, parents=False)

    results = torch.zeros((len(image_dataset), len(YOGO_CLASS_ORDERING) + 5, Sy, Sx))
    for i, (img_batch, fnames) in enumerate(
        tqdm(
            image_dataloader, disable=not use_tqdm, unit_scale=batch_size, unit="images"
        )
    ):
        res = model(img_batch.to(device)).to("cpu", non_blocking=True)

        if output_dir is not None and not draw_boxes:
            out_fnames = [
                Path(output_dir) / Path(fname).with_suffix(".txt").name
                for fname in fnames
            ]
            save_preds(out_fnames, res, thresh=0.5, label=label)
        elif draw_boxes:
            for img_idx in range(img_batch.shape[0]):
                bbox_img = draw_yogo_prediction(
                    img_batch[img_idx, ...],
                    res[img_idx, ...],
                    thresh=0.5,
                    labels=YOGO_CLASS_ORDERING,
                    images_are_normalized=cfg["normalize_images"],
                )
                if output_dir is not None:
                    out_fname = (
                        Path(output_dir)
                        / Path(fnames[img_idx]).with_suffix(".png").name
                    )
                    # mypy thinks that you can't save a PIL Image which is false
                    bbox_img.save(out_fname)  # type: ignore
                else:
                    fig, ax = plt.subplots()
                    ax.set_axis_off()
                    ax.imshow(bbox_img)
                    plt.show()
        elif print_results:
            print(res)
        else:
            # sometimes we return a number of images less than the batch size,
            # namely when len(image_dataset) % batch_size != 0
            results[i * batch_size : i * batch_size + res.shape[0], ...] = res.cpu()

    if not print_results and output_dir is None:
        return results
    return None


def do_infer(args):
    predict(
        args.pth_path,
        path_to_images=args.path_to_images,
        path_to_zarr=args.path_to_zarr,
        output_dir=args.output_dir,
        draw_boxes=args.draw_boxes,
        batch_size=args.batch_size,
        use_tqdm=(args.output_dir is not None or args.draw_boxes),
    )


if __name__ == "__main__":
    parser = infer_parser()
    args = parser.parse_args()
    do_infer(args)
