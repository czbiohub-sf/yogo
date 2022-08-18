# you only glance once

# Assumed structure of dataset folder:


    description.yaml

    somewhere else...
      images/
        image1.png
        image2.png
        ...
      labels/
        image1.csv
        image2.csv
        ...


In the "labels" folder, each text file corresponds to one image file in "images", with
the image extension replaced with "txt". The files "train.txt", "val.txt", and "test.txt"
each have a list of image files that are in their respective datasets. They partition the
total dataset - so train ∪ val ∪ test = dataset, and train ∩ val = 0, train ∩ test = 0,
and val ∩ test = 0.

Each label file should be a CSV (with or without a header) with the following columns in the following order:

`class_index,x_center,y_center,width,height`

class_index is the 0-index of the class from the `class_names` field of `description.yaml` (see [below](#descriptionyaml-requirements`)). `x_center` and `width` are normalized to the width of the image, and `y_center` and `height` are normalized to the height of the image.

## `description.yaml` requirements

The file `description.yaml` describes the class to label map. E.G.

```yaml
class_names: ["healthy", "ring", "schitzont", "troph"]
image_path: <absolute_path_to_images>
label_path: <absolute_path_to_labels>
dataset_split_fractions:
  train: 0.7
  test:  0.25
  val:   0.05
```


## Next steps

- dataloader + training loop
- network finetuning for speed / results
