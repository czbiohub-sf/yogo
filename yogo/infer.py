#! /usr/bin/env python3

import zarr
import math
import torch
import signal

import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from pathlib import Path
from typing import Sequence, TypeVar, List, Union, Optional

from torchvision.transforms import Resize, Compose

from yogo.model import YOGO
from yogo.utils.argparsers import infer_parser
from yogo.utils import draw_rects, format_preds, iter_in_chunks
from yogo.data.dataset import read_grayscale, YOGO_CLASS_ORDERING


# lets us ctrl-c to exit while matplotlib is showing stuff
signal.signal(signal.SIGINT, signal.SIG_DFL)


T = TypeVar("T")


def argmax(arr):
    return max(range(len(arr)), key=arr.__getitem__)


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_preds(fnames, batch_preds, thresh=0.5):
    bs, Sy, Sx, pred_shape = batch_preds.shape

    for fname, batch_pred in zip(fnames, batch_preds):
        preds = format_preds(batch_pred)

        pred_string = "\n".join(
            f"{argmax(pred[5:])} {pred[0]} {pred[1]} {pred[2]} {pred[3]}"
            for pred in preds
        )
        with open(fname, "w") as f:
            f.write(pred_string)


class ImageLoader:
    def __init__(self, _iter, _num_els):
        self._iter = _iter
        self._num_els = _num_els

    def __iter__(self):
        return self._iter()

    def __len__(self):
        if self._iter is None:
            raise RuntimeError(
                "instantiate ImageLoader with `load_image_data` or `load_zarr_data`"
            )
        return self._num_els

    @staticmethod
    def create_batch_from_fnames(
        fnames: Sequence[Union[str, Path]],
        transform: nn.Module,
        normalize_images: bool = False,
        device: Union[str, torch.device] = "cpu",
    ):
        img_batch = torch.stack([read_grayscale(str(fname)) for fname in fnames])
        img_batch = img_batch.to(device)
        img_batch = transform(img_batch)

        if len(img_batch.shape) == 3:
            img_batch.unsqueeze_(dim=0)

        if normalize_images:
            img_batch /= 255

        return img_batch

    @classmethod
    def load_image_data(
        cls,
        path_to_data: Path,
        transform_list: List[nn.Module] = [],
        n: int = 1,
        fnames_only: bool = False,
        normalize_images: bool = False,
        device: Union[str, torch.device] = "cpu",
    ):
        "takes a path to either a single png image or a folder of pngs"
        transform = Compose(transform_list)

        data = (
            [path_to_data]
            if path_to_data.is_file()
            else list(path_to_data.glob("*.png"))
        )

        _num_els = len(data)

        def _iter():
            for fnames in iter_in_chunks(sorted(data), n):
                if fnames_only:
                    yield fnames
                else:
                    yield cls.create_batch_from_fnames(
                        fnames,
                        transform=transform,
                        normalize_images=normalize_images,
                        device=device,
                    )

        return cls(_iter, _num_els)

    @classmethod
    def load_zarr_data(
        cls,
        path_to_zarr: Path,
        transform_list: List[nn.Module] = [],
        n: int = 1,
        normalize_images: bool = False,
        device: Union[str, torch.device] = "cpu",
    ):
        zarr_store = zarr.open(str(path_to_zarr), mode="r")
        transform = Compose([torch.Tensor, *transform_list])

        _num_els = zarr_store.initialized

        def _iter():
            for rg in iter_in_chunks(range(_num_els), n):
                img_batch = zarr_store[:, :, rg.start : rg.stop].transpose((2, 0, 1))
                img_batch = transform(img_batch)
                img_batch.unsqueeze_(dim=1)
                if normalize_images:
                    img_batch /= 255
                yield img_batch.to(device)

        return cls(_iter, _num_els)


def predict(
    path_to_pth: str,
    path_to_images: Optional[Path] = None,
    path_to_zarr: Optional[Path] = None,
    output_dir: Optional[str] = None,
    thresh: float = 0.5,
    draw_boxes: bool = False,
    batch_size: int = 16,
    use_tqdm: bool = False,
):
    if draw_boxes:
        batch_size = 1

    device = choose_device()

    model, cfg = YOGO.from_pth(Path(path_to_pth), inference=True)
    model.to(device)

    img_h, img_w = model.get_img_size()

    normalize_images = cfg["normalize_images"]
    R = Resize([img_h, img_w])

    if path_to_images is not None and path_to_zarr is not None:
        raise ValueError(
            "can only take one of 'path_to_images' or 'path_to_zarr', but got both"
        )
    elif path_to_images is not None:
        image_loader = ImageLoader.load_image_data(
            path_to_images,
            transform_list=[R],
            n=batch_size,
            fnames_only=(output_dir is not None),
            normalize_images=normalize_images,
            device=device,
        )
    elif path_to_zarr is not None:
        image_loader = ImageLoader.load_zarr_data(
            path_to_zarr,
            transform_list=[R],
            n=batch_size,
            normalize_images=normalize_images,
            device=device,
        )
    else:
        raise ValueError("one of 'path_to_images' or 'path_to_zarr' must not be None")

    if not use_tqdm:

        def tqdm_(v):
            return v

    else:
        tqdm_ = tqdm

    N = int(math.log(len(image_loader), 10) + 1)
    for i, data in enumerate(tqdm_(image_loader)):
        if isinstance(data, torch.Tensor):
            fnames = [f"img_{i*batch_size + j:0{N}}" for j in range(batch_size)]
            img_batch = data
            res = model(img_batch)
        else:
            # data is a list of filenames, so we have to create the batch here
            fnames = data
            img_batch = ImageLoader.create_batch_from_fnames(
                fnames, transform=R, device=device
            )
            res = model(img_batch)

        if output_dir is not None and not draw_boxes:
            out_fnames = [
                Path(output_dir) / Path(fname).with_suffix(".txt").name
                for fname in fnames
            ]
            save_preds(out_fnames, res, thresh=0.5)
        elif draw_boxes:
            # batch size is 1 (see above)
            drawn_img = draw_rects(
                img_batch[0, ...], res[0, ...], thresh=0.5, labels=YOGO_CLASS_ORDERING
            )
            if output_dir is not None:
                out_fname = (
                    Path(output_dir) / Path(fnames.pop()).with_suffix(".png").name
                )
                drawn_img.save(out_fname)
            else:
                fig, ax = plt.subplots()
                ax.set_axis_off()
                ax.imshow(drawn_img, cmap="gray")
                plt.show()
        else:
            print(res)


def do_infer(args):
    predict(
        args.pth_path,
        path_to_images=args.path_to_images,
        path_to_zarr=args.path_to_zarr,
        output_dir=args.output_dir,
        draw_boxes=args.draw_boxes,
        use_tqdm=(args.output_dir is not None or args.draw_boxes),
    )


if __name__ == "__main__":
    parser = infer_parser()
    args = parser.parse_args()
    do_infer(args)
