#! /usr/bin/env python3

import torch
import signal

from pathlib import Path

import matplotlib.pyplot as plt

from pathlib import Path

from tqdm import tqdm
from torchvision.transforms import Resize

from yogo.model import YOGO
from yogo.utils import draw_rects
from yogo.argparsers import infer_parser
from yogo.dataloader import read_grayscale


# lets us ctrl-c to exit while matplotlib is showing stuff
signal.signal(signal.SIGINT, signal.SIG_DFL)


def argmax(arr):
    return max(range(len(arr)), key=arr.__getitem__)


def save_preds(fname, batch_preds, thresh=0.5):
    bs, pred_shape, Sy, Sx = batch_preds.shape
    if bs != 1:
        raise ValueError(
            f"can only recieve batch size of 1 (for now) - batch size {bs}"
        )

    reformatted_preds = batch_preds[0, ...].view(pred_shape, Sx * Sy).T
    objectness_mask = (reformatted_preds[:, 4] > thresh).bool()
    preds = reformatted_preds[objectness_mask]
    pred_string = "\n".join(
        f"{argmax(pred[5:])},{pred[0]},{pred[1]},{pred[2]},{pred[3]}"
        for pred in preds
    )
    with open(fname, "w") as f:
        f.write(pred_string)


def predict(
    path_to_pth: str,
    path_to_images: str,
    output_dir: str,
    thresh: float = 0.5,
    visualize: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pth = torch.load(path_to_pth, map_location="cpu")
    img_h, img_w = pth["model_state_dict"]["img_size"]
    model, _ = YOGO.from_pth(Path(path_to_pth), inference=True)
    R = Resize([img_h, img_w])
    model.to(device)

    data = Path(path_to_images)
    if data.is_dir():
        imgs = [str(d) for d in data.glob("*.png")]
    else:
        imgs = [str(data)]

    for fname in imgs:
        img = R(read_grayscale(fname))
        res = model(img[None, ...].to(device))

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
            "output_dir is not set and --visualize flag is not present - nothing to do"
        )
    predict(args.pth_path, args.images, args.output_dir, visualize=args.visualize)


if __name__ == "__main__":
    parser = infer_parser()
    args = parser.parse_args()
    do_infer(args)
