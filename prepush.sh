#! /usr/bin/bash

mypy .
ruff . --fix
black .
python3 -m unittest discover .
