#! /usr/bin/env python3


from pathlib import Path

from labelling_constants import IMG_WIDTH, IMG_HEIGHT

from label_studio_converter.imports.yolo import convert_yolo_to_ls


def generate_tasks_for_runset(path_to_runset_folder: Path):
    folders = [Path(p).parent for p in path_to_runset_folder.glob("./**/labels")]

    def tqdm(v):
        return v
    if __name__ == "__main__":
        try:
            from tqdm import tqdm  # type: ignore
        except ImportError:
            print("missing tqdm")
            pass
    else:
        print("not main")

    for folder_path in tqdm(folders):
        if not folder_path.is_dir():
            continue

        try:
            convert_yolo_to_ls(
                input_dir=str(folder_path),
                out_file=str(folder_path / "tasks.json"),
                out_type="predictions",
                image_root_url="http://localhost:8081",
                image_ext=".png",
                image_width=IMG_WIDTH,
                image_height=IMG_HEIGHT,
            )
        except PermissionError:
            print(f"permission error for file {folder_path}. continuing...")
            continue


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path to runset>")
        sys.exit(1)

    path_to_runset = Path(sys.argv[1])

    if not path_to_runset.exists():
        raise ValueError(f"{str(path_to_runset)} doesn't exist")

    generate_tasks_for_runset(path_to_runset)
