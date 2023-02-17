#! /usr/bin/env python3

import sys
import signal

import matplotlib.pyplot as plt

from pathlib import Path
from typing import Iterable, Tuple, List

from yogo.utils import draw_rects
from yogo.dataloader import load_labels_from_path, read_grayscale

from labelling_constants import CLASSES

signal.signal(signal.SIGINT, signal.SIG_DFL)


def find_label_file(label_dir: Path, image_path: Path) -> Path:
    extensions = (".csv", ".txt", ".tsv", "")
    for ext in extensions:
        label_path = label_dir / image_path.with_suffix(ext).name
        if label_path.exists():
            return label_path

    raise FileNotFoundError(f"label file not found for {str(image_path)}")


def plot_img_labels_pair(image_path: Path, label_path: Path):
    labels = load_labels_from_path(label_path, classes=range(len(CLASSES)))

    img = read_grayscale(str(image_path)).squeeze()
    annotated_img = draw_rects(img, labels)

    plt.imshow(annotated_img)
    plt.show()


def make_img_label_pairs(image_dir: Path, label_dir: Path) -> List[Tuple[Path, Path]]:
    if label_dir.is_dir():
        image_label_pairs = []
        # get an iterable from `image_dir` - if it is a directory, find all pngs.
        # if its a file, we will try to find it's label, so just throw it in a list.
        if image_dir.is_dir():
            image_iter: Iterable = image_dir.glob("*.png")
        elif image_dir.is_file():
            image_iter = [image_dir]
        else:
            raise ValueError(
                f"provided argument image_dir is neither a file nor a directory: {image_dir}"
            )

        for image_path in image_iter:
            try:
                label_path = find_label_file(label_dir, image_path)
                image_label_pairs.append((image_path, label_path))
            except FileNotFoundError as e:
                print(f"no label file: {e}")
                print("continuing...")
                continue

        return image_label_pairs
    elif image_dir.is_file() and label_dir.is_file():
        return [(image_dir, label_dir)]

    raise ValueError(
        "image dir and label dir are invalid (i.e. not directories nor files):\n"
        f"\timage_dir={image_dir}\n"
        f"\tlabel_dir={label_dir}"
    )


if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print(
            f"usage: {sys.argv[0]} <path to image or image folder> [<path to label or label folder>]"
        )
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    if len(sys.argv) == 3:
        label_dir = Path(sys.argv[2])
    else:
        if image_dir.name == "images":
            label_dir = image_dir.parent / "labels"
        elif image_dir.parent.name == "images":
            label_dir = image_dir.parent.parent / "labels"

    image_label_pairs = make_img_label_pairs(image_dir, label_dir)

    for image_path, label_path in image_label_pairs:
        plot_img_labels_pair(image_path, label_path)
