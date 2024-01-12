#! /usr/bin/env bash

mypy .
ruff . --fix
black .
pytest -q tests/
