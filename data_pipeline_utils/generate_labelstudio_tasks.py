#! /usr/bin/env python3


from pathlib import Path
from typing import Union, Optional
from urllib.request import pathname2url

from labelling_constants import IMG_WIDTH, IMG_HEIGHT, IMAGE_SERVER_PORT

from label_studio_converter.imports.yolo import convert_yolo_to_ls


def tqdm(v):
    return v


if __name__ == "__main__":
    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        pass


def path_is_relative_to(path_a: Path, path_b: Union[str, Path]) -> bool:
    """
    Path.is_relative_to is available in pathlib since 3.9,
    but we are running 3.7. Copied from pathlib
    (https://github.com/python/cpython/blob/main/Lib/pathlib.py)
    """
    path_b = type(path_a)(path_b)
    return path_a == path_b or path_b in path_a.parents


def path_relative_to(path_a: Path, path_b: Union[str, Path], walk_up=False) -> Path:
    """
    Path.relative_to is available in pathlib since 3.9,
    but we are running 3.7. Copied from pathlib
    (https://github.com/python/cpython/blob/main/Lib/pathlib.py)
    """
    path_cls = type(path_a)
    path_b = path_cls(path_b)

    for step, path in enumerate([path_b] + list(path_b.parents)):
        if path_is_relative_to(path_a, path):
            break
    else:
        raise ValueError(f"{str(path_a)!r} and {str(path_b)!r} have different anchors")

    if step and not walk_up:
        raise ValueError(f"{str(path_a)!r} is not in the subpath of {str(path_b)!r}")

    parts = ("..",) * step + path_a.parts[len(path.parts) :]
    return path_cls(*parts)


def generate_tasks_for_runset(path_to_runset_folder: Path):
    folders = [Path(p).parent for p in path_to_runset_folder.glob("./**/labels")]

    if len(folders) == 0:
        raise ValueError(
            f"couldn't find labels and images - double check the provided path"
        )

    for folder_path in tqdm(folders):
        if not folder_path.is_dir():
            print(f"warning: {folder_path} is not a directory")
            continue

        abbreviated_path = str(path_relative_to(folder_path, path_to_runset_folder))
        root_url = f"http://localhost:{IMAGE_SERVER_PORT}/{pathname2url(abbreviated_path)}/images"

        tasks_path = str(folder_path / "tasks.json")

        try:
            convert_yolo_to_ls(
                input_dir=str(folder_path),
                out_file=tasks_path,
                out_type="predictions",
                image_root_url=root_url,
                image_ext=".png",
                image_dims=(IMG_WIDTH, IMG_HEIGHT),
            )
        except TypeError:
            # we aren't using our custom version, so try default
            print(
                "warning: couldn't give convert_yolo_to_ls image dims, so defaulting "
                "to slow version"
            )
            convert_yolo_to_ls(
                input_dir=str(folder_path),
                out_file=tasks_path,
                out_type="predictions",
                image_root_url=root_url,
                image_ext=".png",
            )
        except Exception as e:
            print(f"exception found for file {folder_path}: {e}. continuing...")
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
