#! /usr/bin/env python3


from pathlib import Path

from label_studio_converter.imports.yolo import convert_yolo_to_ls


def generate_tasks_for_runset(path_to_runset_folder: Path):
    folders = [Path(p).parent for p in path_to_runset_folder.glob(f"./**/labels")]

    for folder_path in folders:
        convert_yolo_to_ls(
            input_dir=str(folder_path),
            out_file=str(folder_path / "tasks.json"),
            out_type="predictions",
            image_root_url="http://localhost:8081",
            image_ext=".png",
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path to runset>")
        sys.exit(1)

    path_to_runset = Path(sys.argv[1])

    if not path_to_runset.exists():
        raise ValueError(f"{str(path_to_runset)} doesn't exist")

    generate_tasks_for_runset(path_to_runset)
