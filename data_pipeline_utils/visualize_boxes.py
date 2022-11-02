import sys
import signal

import matplotlib.pyplot as plt

from pathlib import Path

from yogo.utils import draw_rects
from yogo.dataloader import load_labels_from_path, read_grayscale


signal.signal(signal.SIGINT, signal.SIG_DFL)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("warning: this tool is quite specific")
        print(f"usage: {sys.argv[0]} <path to image folder> <path to label folder>")
        sys.exit(1)

    images = Path(sys.argv[1])
    labels = Path(sys.argv[2])
    for image in images.glob("*.png"):
        label_path = labels / image.with_suffix(".csv").name
        labels = load_labels_from_path(label_path)
        img = read_grayscale(image)
        annotated_img = draw_rects(img, labels)

        plt.imshow(annotated_img)
        plt.show()
