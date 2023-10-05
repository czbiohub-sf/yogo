# you only glance once

A version of the [YOLO architecture (versions 1 through 3)](https://pjreddie.com/darknet/yolo/), optimized for speed on slow hardware and simple object detection problems.

## Install

```console
python3 -m pip install -e .
```

If you want to export to onnx,
```console
python3 -m pip install -e ".[onnx]"
```

And finally, if you want to export a model to Openvino's format, you will also need to install [their code](https://docs.openvino.ai/2023.0/openvino_docs_install_guides_overview.html?ENVIRONMENT=DEV_TOOLS&OP_SYSTEM=LINUX&VERSION=v_2023_0_2&DISTRIBUTION=PIP).

## YOGO Docs

Documentation for YOGO. If you want documentation in a specific area, [let me know](https://github.com/czbiohub-sf/yogo/issues/new)!

- [recipes.md](docs/recipes.md) has the basics of using YOGO in your own code
- [cli.md](docs/cli.md) is a short guide on how to use YOGO from the command line (via `yogo`)
- [yogo-high-level.md](docs/yogo-high-level.md) is a high level guide of the YOGO architecture
- [training.md](docs/training.md) has some basic information about training YOGO
- [dataset-definition.md](docs/dataset-definition.md) defines the dataset description files, the files YOGO uses to define datasets for training

Also, see my talk discussing how we interpret parasite classifications from YOGO prediction tensors [here](https://drive.google.com/file/d/1S5UZEGtEtVwHKBzKIGvCUJlRAFgw-b9H/view) (start around 24:00)
