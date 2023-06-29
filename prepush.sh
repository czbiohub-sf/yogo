#! /usr/bin/bash

mypy .
ruff . --fix
black .
