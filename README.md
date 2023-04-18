# you only glance once

## What

A version of the [YOLO architecture (versions 1 through 3)](https://pjreddie.com/darknet/yolo/), optimized for malaria detection on the ULC-Malaria-Scope.

## Why

- Need to count the number of malarial vs. healthy red blood cells in a patient sample with high accuracy
- Need to count at (almost) real-time (around 30 FPS) on restrictive hardware (Raspberry Pi 4 + Intel Neural Compute Stick 2)
- Existing YOLOv\* models run very slowly on the Pi (300 ms+ inference, or about 3 FPS)

## Installation

You will need to install PyTorch and TorchVision. Go [to PyTorch's website](https://pytorch.org/get-started/locally/) and follow their instructions. Then,

```console
python3 -m pip install -e .
```

You can also install YOGO with any/all of the options below:

### Installation for Training

If you want to export models, run

```console
python3 -m pip install ".[onnx]"
```

## Training

Train locally by running

```console
yogo train <path to dataset_definition.yml> [opts]
```
for list of opts, run `yogo train --help`.

To train on SLURM, run

```console
sbatch submit_cmd.sh yogo train <path to dataset_definition.yml> [opts]
```
with the same options from above.

To run a sweep on SLURM, first modify `sweep.yml` to fit your needs. Then, run

```console
wandb sweep sweep.yml
```

which should give you a sweep ID that looks like `bioengineering/yogo/foo`. Then start each sweep job by running

```console
sbatch sweep_launch.sh bioengineering/yogo/foo
```

## Exporting

To export a model, make sure you installed `yogo` with Onnx, and run

```console
yogo export <model pth file>
```

You can run `yogo export --help` to get the full list of options. The defaults should be pretty good though!
