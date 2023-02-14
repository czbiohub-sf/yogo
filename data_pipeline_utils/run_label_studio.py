#! /usr/bin/env python3


import os

from pathlib import Path
from multiprocessing import Process
from http.server import HTTPServer, ThreadingHTTPServer, SimpleHTTPRequestHandler


def run_server(directory):
    server_addy = ("localhost", 8081)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

    httpd = ThreadingHTTPServer(server_addy, Handler)

    print('going to serve')
    httpd.serve_forever()


# run_server(".")


def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler):
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()

run()

"""
cd "$INPUT_DIR" && python3 -m http.server 8081


os.environ["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = True
os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"]="$INPUT_DIR"
"""
