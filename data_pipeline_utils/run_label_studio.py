#! /usr/bin/env python3


import os
import argparse
import subprocess

from pathlib import Path
from multiprocessing import Process
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler


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
        type=Path,
        help="path to run folder (i.e. folder containing 'images' and 'labels'",
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
    from labelling_constants import FLEXO_DATA_DIR

    os.environ["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = FLEXO_DATA_DIR

    proc = run_server_in_proc(Path(FLEXO_DATA_DIR))

    try:
        subprocess.run(["label-studio", "start"])
    except KeyboardInterrupt:
        print("gee wiz, thank you for labelling today!")
