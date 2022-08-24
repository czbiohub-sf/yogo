#! /usr/bin/env python3

import sys
import torch
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize

from pathlib import Path

from model import YOGO



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <path_to_pth> <path_to_image_or_images>")

    pth = torch.load(sys.argv[1], map_location=torch.device('cpu'))

    m = YOGO(17,17)
    m.load_state_dict(pth["model_state_dict"])
    m.eval()

    img_h, img_w = 300, 400
    R = Resize([img_h,img_w])

    data = Path(sys.argv[2])
    if data.is_dir():
        raise NotImplementedError()
        # for img_path in data.glob("*.png"):
        #     img = R(read_image(str(img_path), ImageReadMode.GRAY))
        #     plt.imshow(img[0,...], cmap='gray')  # imshow doesn't like C=1 for CHW imgs
        #     plt.show()
    else:
        img = R(read_image(str(data), ImageReadMode.GRAY))
        fig, ax = plt.subplots()

        ax.imshow(img[0,...], cmap='gray')  # imshow doesn't like C=1 for CHW imgs

        res = m(img[None, ...])
        _, pred_dim, Sy, Sx = res.shape
        for pred in torch.permute(res.reshape(1, pred_dim, Sx * Sy)[0,:,:], (1,0)):
            assert len(pred) == 9
            xc, yc, w, h = pred[:4]
            xc, yc, w, h = xc.item(), yc.item(), w.item(), h.item()
            print(xc, yc, w, h)
            ax.add_patch(
                Rectangle(
                    (img_w * (xc - w / 2), img_h * (yc - h / 2)),
                    img_w * w,
                    img_h * h,
                    facecolor="none",
                    edgecolor="black",
                )
            )


        plt.show()
