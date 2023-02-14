#! /usr/bin/env python3

import sys
import signal

import matplotlib.pyplot as plt

from pathlib import Path

from yogo.utils import draw_rects
from yogo.dataloader import load_labels_from_path, read_grayscale


signal.signal(signal.SIGINT, signal.SIG_DFL)


def find_label_file(label_dir: Path, img_path: Path) -> Path:
    extensions = (".csv", ".txt", ".tsv", "")
    for ext in extensions:
        label_path = label_dir / image_path.with_suffix(ext).name
        if label_path.exists():
            return label_path

    raise FileNotFoundError(f"label file not found for {str(img_path)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("warning: this tool is quite specific")
        print(f"usage: {sys.argv[0]} <path to image folder> <path to label folder>")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    label_dir = Path(sys.argv[2])
    for image_path in image_dir.glob("*.png"):
        try:
            label_path = find_label_file(label_dir, image_path)
        except FileNotFoundError as e:
            print(f"no label file: {e}")
            print("continuing...")
            continue

        labels = load_labels_from_path(label_path, classes=range(4))

        img = read_grayscale(str(image_path)).squeeze()
        annotated_img = draw_rects(img, labels)

        plt.imshow(annotated_img)
        plt.show()
