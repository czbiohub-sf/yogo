#! /usr/bin/env python3

import sys
import torch
import signal

# lets us ctrl-c to exit while matplotlib is showing stuff
signal.signal(signal.SIGINT, signal.SIG_DFL)

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Optional

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize

from model import YOGO
from utils import draw_rects


if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print(f"usage: {sys.argv[0]} [<path_to_pth>] <path_to_image_or_images>")
        sys.exit(1)

    if len(sys.argv) == 3:
        pth = torch.load(sys.argv[1], map_location=torch.device("cpu"))
        img_h, img_w = pth["model_state_dict"]["img_size"]
        model = YOGO.from_pth(pth, inference=True)
        data_path = sys.argv[2]
    else:
        img_h, img_w = 600, 800
        model = YOGO((img_h, img_w), 0.01, 0.01, inference=True)
        data_path = sys.argv[1]

    R = Resize([img_h, img_w])

    data = Path(data_path)
    if data.is_dir():
        imgs = [str(d) for d in data.glob("*.png")]
    else:
        imgs = [str(data)]

    for fname in imgs:
        print(fname)
        img = R(read_image(fname, ImageReadMode.GRAY))
        fig, ax = plt.subplots()

        res = model(img[None, ...])
        drawn_img = draw_rects(img[0, ...], res[0, ...], thresh=0.5)
        ax.imshow(drawn_img, cmap="gray")

        plt.show()
