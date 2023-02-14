#! /usr/bin/env python3


import os
import sys
import subprocess

from pathlib import Path
from multiprocessing import Process
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler


def run_server(directory):
    port = 8081
    server_addy = ("localhost", 8081)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

    httpd = ThreadingHTTPServer(server_addy, Handler)

    print(f"serving your files, Hot n' Fresh, from http://localhost:{port}")
    httpd.serve_forever()


def run_server_in_proc(directory) -> Process:
    p = Process(target=run_server, args=("../..",), daemon=True)
    p.start()
    return p


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path to runset folder>")
        sys.exit(1)

    path_to_runset_folder = Path(sys.argv[1])

    if not path_to_runset_folder.exists():
        raise ValueError(f"{str(path_to_runset_folder)} doesn't exist")

    os.environ["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(path_to_runset_folder)

    proc = run_server_in_proc(path_to_runset_folder)

    subprocess.run(["label-studio", "start"])
