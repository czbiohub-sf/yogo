#! /usr/bin/env python3


import os
import argparse
import subprocess

from pathlib import Path
from multiprocessing import Process
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

from labelling_constants import FLEXO_DATA_DIR
from generate_labelstudio_tasks import generate_tasks_for_runset


try:
    boolean_action = argparse.BooleanOptionalAction  # type: ignore
except AttributeError:
    boolean_action = "store_true"  # type: ignore


def get_parser():
    """
    I wanted to add the option to import `tasks.json`, but you need an api key for
    that. Can we create a 'global' api key that can create projects? dunno!
    """
    parser = argparse.ArgumentParser(description="label studio runner!")

    parser.add_argument(
        dest="run_folder",
        metavar="run-folder",
        nargs="?",
        type=Path,
        help=(
            "path to run folder (i.e. folder containing 'images' and 'labels'), "
            "defaults to running on OnDemand if no argument is provided"
        ),
        default=FLEXO_DATA_DIR,
    )
    return parser


def run_server(directory: Path):
    port = 8081
    server_addy = ("localhost", 8081)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

    httpd = ThreadingHTTPServer(server_addy, Handler)

    print(
        f"serving your files, Hot n' Fresh, on http://localhost:{port} from {str(directory)}"
    )
    httpd.serve_forever()


def run_server_in_proc(directory: Path) -> Process:
    p = Process(target=run_server, args=(directory,), daemon=True)
    p.start()
    return p


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    path_to_run_folder = args.run_folder

    os.environ["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(path_to_run_folder)

    proc = run_server_in_proc(path_to_run_folder)

    if path_to_run_folder != Path(FLEXO_DATA_DIR):
        # create tasks.json here
        generate_tasks_for_runset(path_to_run_folder, task_folder_path=Path("."))
        print(f"tasks file written to {str(Path('./tasks.json').absolute())}")

    try:
        subprocess.run(["label-studio", "start"])
    except KeyboardInterrupt:
        print("gee wiz, thank you for labelling today!")
