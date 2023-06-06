# Dataset Definition Files

There are many different conditions over which we may want to train - e.g. only fast flowrates, only human-annotated labels, e.t.c. Therefore we would like
a dataset definition format for YOGO that is easily machine-readable (so YOGO can process it) and human-readable (so we can edit the files by hand, if we wish).

Dataset definition files are central to training: it is the first argument you pass to YOGO when starting a training run (i.e. `yogo train dataset_defn.yml`). They are also generated when you label data via our [labelling scripts](https://github.com/czbiohub/lfm-data-utilities/blob/main/lfm_data_utilities/malaria-labelling/scripts.md#creating-cellpose-or-yogo-labels).

## Dataset Definition file requirements

The file `example_dataset_description.yml` describes the class to label map. E.G.

```yaml
# DATASET DESCRIPTION FILE
#
# Here, we are describing our dataset. There are only a couple pieces
# of information that you have to supply. The first is the class names:
class_names:
  - healthy
  - ring
  - trophozoite
  - schizont
  - gametocyte
  - wbc
  - misc

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

# You can also specify test paths. The data specified here will be isolated
# from training data, so we can get proper measurements of model quality.
# If you specify `test_paths`, the `dataset_split_fractions` section above
# can only have `test` and `val` as keys, and remember that their values must
# still sum to 1. E.g.,
#
# dataset_split_fractions:
#   test:  0.75
#   val:   0.25
#
test_paths:
  set1:  # this name is just for your convenience!
    image_path: /path/to/images/
    label_path: /path/to/labels/
  ...
 
If you want, you can also augment certain classes by human bbox labels!
thumbnail_agumentation:
  misc: /path/to/folder/of/toner/blob/thumbnails
  wbc: /path/to/folder/of/wbc/thumbnails
  misc: /another/path/to/misc
```

## Label files

In each "labels" folder, each text file corresponds to one image file in "images", with the image extension replaced with "txt". Each label file should be a CSV (with or without a header) with the following columns in the following order:

```
class_index,x_center,y_center,width,height  # each of these rows describe one bounding box
class_index,x_center,y_center,width,height
class_index,x_center,y_center,width,height
```

`class_index` is the 0-index of the class from the `class_names` field of `description.yaml`. `x_center` and `width` are normalized to the width of the image, and `y_center` and `height` are normalized to the height of the image.
