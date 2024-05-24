# you only glance once

A version of the [YOLO architecture (versions 1 through 3)](https://pjreddie.com/darknet/yolo/), optimized for speed on slow hardware and simple object detection problems. Designed for the Remoscope project, by the Bioengineering team at the [Chan-Zuckerberg Biohub SF](https://www.czbiohub.org/sf/).

```console
$ yogo train path/to/dataset-definition-file.yml  # train your model!
$ yogo infer path/to/model.pth  # use your model!
$ yogo export path/to/model.pth  # use your model somewhere else!
$ yogo test path/to/model.pth path/to/dataset-definition-file. # test your model!
$ yogo --help  # all the other details are here :)
```

## Install

```console
python3 -m pip install -e .
```

> [!NOTE]
> Openvino and M1s do not play together very well. If exporting to Openvino's format, I suggest using a VM or Docker.

## Docs

Documentation for YOGO. If you want documentation in a specific area, [let me know](https://github.com/czbiohub-sf/yogo/issues/new)!

- [recipes.md](docs/recipes.md) has the basics of using YOGO in your own code
- [cli.md](docs/cli.md) is a short guide on how to use YOGO from the command line (via `yogo`)
- [yogo-high-level.md](docs/yogo-high-level.md) is a high level guide of the YOGO architecture
- [training.md](docs/training.md) has some basic information about training YOGO
- [dataset-definition.md](docs/dataset-definition.md) defines the dataset description files, the files YOGO uses to define datasets for training

Also, see my talk discussing how we interpret parasite classifications from YOGO prediction tensors [here](https://drive.google.com/file/d/1S5UZEGtEtVwHKBzKIGvCUJlRAFgw-b9H/view) (start around 24:00)

## Contributing Guidelines

Please run `./prepush.sh` before pushing. It runs [`mypy`](https://mypy-lang.org/), [`ruff`](https://docs.astral.sh/ruff/), [`black`](https://github.com/psf/black) and [`pytest`](https://docs.pytest.org/en/8.2.x/).

When creating issues or pull requests, please be detailed. What exact commands were you running on what computer to get your issue? What exactly does your PR contribute and why is it necessary?
