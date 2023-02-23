#! /usr/bin/env python3

import zarr
import torch
import signal

# lets us ctrl-c to exit while matplotlib is showing stuff
signal.signal(signal.SIGINT, signal.SIG_DFL)

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from torchvision.transforms import Resize

from .model import YOGO
from .utils import draw_rects
from .argparsers import infer_parser
from .dataloader import read_grayscale


def argmax(arr):
    return max(range(len(arr)), key=arr.__getitem__)


def save_preds(fname, res, thresh=0.5):
    bs, pred_dim, Sy, Sx = res.shape
    if bs != 1:
        raise ValueError(
            f"can only recieve batch size of 1 (for now) - batch size {bs}"
        )

    with open(fname, "w") as f:
        for j in range(Sy):
            for i in range(Sx):
                pred = res[0, :, j, i]
                # if objectness t0 is greater than threshold
                if pred[4] > 0.5:
                    f.write(
                        f"{argmax(pred[5:])},{pred[0]},{pred[1]},{pred[2]},{pred[3]}\n"
                    )


class ImageLoader:
    def __init__(self, _iter, _num_els):
        self._iter = _iter
        self._num_els = _num_els

    def __iter__(self):
        if self._iter is None:
            raise RuntimeError(
                "instantiate ImageLoader with `load_image_data` or `load_zarr_data`"
            )

        return self._iter()

    def __len__(self):
        if self._iter is None:
            raise RuntimeError(
                "instantiate ImageLoader with `load_image_data` or `load_zarr_data`"
            )

        return self._num_els

    @classmethod
    def load_image_data(cls, path_to_data: str):
        "takes a path to either a single png image or a folder of pngs"
        datapath = Path(path_to_data)
        data = [datapath] if datapath.is_file() else datapath.glob("*.png")

        def _iter():
            for img_name in sorted(data):
                yield cv2.imread(str(img_name), cv2.IMREAD_GRAYSCALE)

        _num_els = 1 if datapath.is_file() else sum(1 for _ in datapath.glob("*.png"))

        return cls(_iter, _num_els)

    @classmethod
    def load_zarr_data(cls, path_to_zarr: str):
        data = zarr.open(path_to_zarr)

        def _iter():
            for i in range(data.initialized):
                yield data[...,i]

        _num_els = len(data)

        return cls(_iter, _num_els)

    @classmethod
    def load_random_data(cls, image_shape, n_iters):
        if len(image_shape) == 2:
            image_shape = (1, 1, *image_shape)
        else:
            raise ValueError(f"image shape must be (h,w) - got {image_shape}")

        rand_tensor = np.random.randn(image_shape)

        def _iter():
            for _ in range(n_iters):
                yield rand_tensor

        _num_els = n_iters

        return cls(_iter, _num_els)

def predict(
    path_to_pth: str,
    image_loader: ImageLoader,
    output_dir: str,
    thresh: float = 0.5,
    visualize: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pth = torch.load(path_to_pth, map_location=device)
    img_h, img_w = pth["model_state_dict"]["img_size"]
    model = YOGO.from_pth(pth, inference=True)

    R = Resize([img_h, img_w])

    for img in tqdm(image_loader):
        img = R(img)
        res = model(img[None, ...])

        if visualize:
            fig, ax = plt.subplots()
            drawn_img = draw_rects(img[0, ...], res[0, ...], thresh=0.5)
            ax.imshow(drawn_img, cmap="gray")
            plt.show()
        else:
            out_fname = Path(output_dir) / Path(fname).with_suffix(".csv").name
            save_preds(out_fname, res, thresh=0.5)


def do_infer(args):
    if args.output_dir is None and not args.visualize:
        raise ValueError(
            f"output_dir is not set and --visualize flag is not present - nothing to do"
        )

    image_data = args.image_data
    if image_data.is_dir() or image_data.suffix == ".png":
        # assume image_data is folder of images
        ImageLoader.load_image_data(image_data)
    elif image_data.suffix == ".zip":
        # assume it is zarr
        ImageLoader.load_zarr_data(image_data)
    else:
        raise ValueError(
            "invalid image_data; it must be a directory of pngs, a png, or a zarr "
            f"file (ending with .zip); got {image_data}"
        )

    predict(args.pth_path, args.images, args.output_dir, visualize=args.visualize)


if __name__ == "__main__":
    parser = infer_parser()
    args = parser.parse_args()
    do_infer(args)
