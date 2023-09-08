# you only glance once

A version of the [YOLO architecture (versions 1 through 3)](https://pjreddie.com/darknet/yolo/), optimized for speed on slow hardware and simple object detection problems.

Here are some [docs](docs/) for YOGO.

To install, you will need to install PyTorch and TorchVision. Go [to PyTorch's website](https://pytorch.org/get-started/locally/) and follow their instructions. Then,

```console
python3 -m pip install -e .
```

If you want to export a model to Openvino's format, you will also need to install [their code](https://docs.openvino.ai/2023.0/openvino_docs_install_guides_overview.html?ENVIRONMENT=DEV_TOOLS&OP_SYSTEM=LINUX&VERSION=v_2023_0_2&DISTRIBUTION=PIP).
