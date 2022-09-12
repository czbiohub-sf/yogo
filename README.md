# you only glance once

[Follow along here!](https://wandb.ai/bioengineering/yogo)

## What

A version of the [YOLO architecture (versions 1 through 3)](https://pjreddie.com/darknet/yolo/), optimized for malaria detection on the ULC-Malaria-Scope.

## Why

- Need to count the number of malarial vs. healthy red blood cells in a patient sample with high accuracy
- Need to count at (almost) real-time (around 30 FPS) on restrictive hardware (Raspberry Pi 4 + Intel Neural Compute Stick 2)
- Existing YOLOv\* models run very slowly on the Pi (300 ms+ inference, or about 3 FPS)

## Results

- Initial benchmark of this model was run at ~40 FPS on the Raspberry Pi 4 w/ NCS2

## Training

Train locally by running

```console
python3 train.py <path to dataset_definition.yml> [opts]
```
for list of opts, run `python3 train.py --help`.

To train on SLURM, run

```console
sbatch yogo_training.sh <path to dataset_definition.yml> [opts]
```
with the same options from `train.py`.

To run a sweep on SLURM, first modify `sweep.yml` to fit your needs. Then, run

```console
wandb sweep sweep.yml
```

which should give you a sweep ID that looks like `bioengineering/yogo/foo`. Then start each sweep job by running

```console
sbatch sweep_launch.sh bioengineering/yogo/foo
```

## Dataset definition

To define the dataset, we use a `.yml` file.

### `description.yml` requirements

The file `example_dataset_description.yml` describes the class to label map. E.G.

```yaml
# DATASET DESCRIPTION FILE
#
# Here, we are describing our dataset. There are only a couple pieces
# of information that you have to supply. The first is the class names:
class_names: ["healthy", "ring", "schitzont", "troph"]
# Neural networks encode classes by integers, and the list above defines this
# ordering by index. I.e. "0" maps to "healthy", "1" maps to ring, e.t.c.
#
# You can define how the dataset is split up. This is the `dataset_split_fractions`
# definition below. It splits up the total dataset (all image-label pairs) by the
# percentages below. Each of these keys are required, and their values must sum
# to 1.
dataset_split_fractions:
  train: 0.7
  test:  0.25
  val:   0.05
# Finally, we have to actually point to our data. If we have just one set of
# folders for images and labels, then we just define the image path and label
# path, like below:
image_path: /path/to/images/
label_path: /path/to/labels/
# However, if we want multiple sets of folders for training and inference, we
# use `dataset_paths` to define the paths to folders individually.
dataset_paths:
  set1:  # this name is just for your convenience!
    image_path: /path/to/images/
    label_path: /path/to/labels/
  set2:
    image_path: /path/to/images/
    label_path: /path/to/labels/
  ...
```

Here is an example file structure for the each of the dataset paths in `example_dataset_description.yml` above.

    /path/to/images/
      image1.png
      image2.png
      ...
    /path/to/labels/
      image1.csv
      image2.csv
      ...

In the "labels" folder, each text file corresponds to one image file in "images", with the image extension replaced with "txt". Each label file should be a CSV (with or without a header) with the following columns in the following order:

  `class_index,x_center,y_center,width,height`

`class_index` is the 0-index of the class from the `class_names` field of `description.yaml`. `x_center` and `width` are normalized to the width of the image, and `y_center` and `height` are normalized to the height of the image.


### TODOs

- ~~Visualize Confusion Tables in W&B~~
- Get output format for scope!
- [Profile training, try to speed it up](https://pytorch.org/tutorials/beginner/profiler.html)
  - profiled, and `chrome://tracing` the `chrome_profile.json` actually is not too bad.
  - Results: yogo_loss.format_labels is SO slow
    - cluster_anchors.centers_to_corners and cluster_anchors.torch_iou are the big offenders
      - fixed by using the tools in torchvision.ops!
      - Naiive multithreading doesn't seem to help - thanks GIL
    - yogo_loss.split_labels_into_bins is a close second
  - loading data is second slowest
  - cuda is DUMB quick, its rediculous - we should try to push more work to the GPU, it is barely used!
- Optimize for model performance
  - Figure out best model structure
  - Correct for class imbalance
  - Data Augmentation (rotations, concatenations)
  - Implement ["Bag of Freebies"](https://arxiv.org/pdf/1902.04103.pdf)
    - Not sure how much is actually applicable - added label smoothing
  - Measure how Sx/Sy vs. output image size affects performance
  - Sweeps for LR / BS /

### Ideas

- How does (Sx, Sy) affect performance? (Sx, Sy) vs. (anchor\_w, anchor\_h)?
- turn `if __name__ ==  __main__` sanity checks into tests
- [conv net ordering of batchnorms, dropouts, maxpools, e.t.c.](https://stackoverflow.com/questions/59634780/correct-order-for-spatialdropout2d-batchnormalization-and-activation-function)

### Weird observations

- Using the M1 chip's MPS accelerator seems to either
  - not converge, basically at all
  - error out with aa note requesting we report to PyTorch
- Why does `dataloader.num_workers = 1` report nothing on W&B? Is it running?
