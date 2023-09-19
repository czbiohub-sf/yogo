#! /usr/bin/env bash

mypy .
ruff . --fix
black .
python3 -m unittest discover .
