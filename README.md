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

## Dataset definition

To define the dataset, we use a `.yml` file.

### `description.yml` requirements

The file `description.yml` describes the class to label map. E.G.

```yaml
class_names: ["healthy", "ring", "schitzont", "troph"]
image_path: <absolute_path_to_image_folder>
label_path: <absolute_path_to_label_folder>
dataset_split_fractions:
  train: 0.7
  test:  0.25
  val:   0.05
```

See `example_dataset_description.yml` for an example dataset description `yaml` file, which will be pretty much a verbatim reproduction of the above.

Here is an example file structure for the `description.yml` above.

    images/
      image1.png
      image2.png
      ...
    labels/
      image1.csv
      image2.csv
      ...

In the "labels" folder, each text file corresponds to one image file in "images", with the image extension replaced with "txt". Each label file should be a CSV (with or without a header) with the following columns in the following order:

  `class_index,x_center,y_center,width,height`

`class_index` is the 0-index of the class from the `class_names` field of `description.yaml`. `x_center` and `width` are normalized to the width of the image, and `y_center` and `height` are normalized to the height of the image.

 `dataset_split_fractions` split up the dataset by those percentages - so in the above example, 70% of the dataset is in the training set, 25% is in the testing set, and 5% is in the validation set. They partition the total dataset - so train ∪ val ∪ test = dataset, and train ∩ val = 0, train ∩ test = 0, and val ∩ test = 0.

### TODOs

- How does (Sx, Sy) affect performance? (Sx, Sy) vs. (anchor\_w, anchor\_h)?
- parameter sweeps?
- solve all `rg "TODO|FIXME" -A 3` (quick summary)
  - figure out PyTorch sync points
  - BAG OF FREEBIESS
  - turn `if __name__ ==  __main__` sanity checks into tests
  - test MPS (why was it converging so poorly?) (see below)
  - profile forward/backward passes
  - look at pinned memory (low probability of success, low priority)

### Miscelaneous Notes

- Using the M1 chip's MPS accelerator seems to either
  - not converge, basically at all
  - error out with aa note requesting we report to PyTorch
