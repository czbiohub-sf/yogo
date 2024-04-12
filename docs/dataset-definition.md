# Dataset Definition Files

I wanted a simple way to represent an object detection dataset for data that is distributed over a file system, without having to copy files or create symlinks, as is common in other image-based deep learning repositories. It is similar to [Ultralytic's dataset definition](https://docs.ultralytics.com/datasets/detect/), except ours is more general, and has a couple features that are very nice to have.

## Basic dataset definition

A "dataset definition" is composed of one or more individual sets, where each set is just an `image_path` to a folder of images, and a `label_path` to a folder of (`.txt`) labels. For each label file in the `label_path` folder, there should be an image file in `image_path`. The should have the same name, except for the extension - so, for label file `img_1729.txt`, there should be a `img_1729.png` (or `.jpg` or whatever) image file in the `image_path` directory[^1].

Each label file follows the standard YOLO label format:

```txt
label_index x_center y_center width height
label_index x_center y_center width height
...
```

where `x_center` and `width`, and the `y_center` and `height` are all normalized to the width and height of the image, respectively.

The class-name-to-label-index map is given by the `class_names` key in the definition file. Note that the label indicies are 0-indexed. So, for

```yaml
...
class_names:
  - dog
  - cat
  - bat
...
```

`dog` would have `label_index` 0, `cat` 1, `bat` 2. An error will be thrown if a label file has an index that doesn't correspond to these class names (e.g. a `label_index` of 5 for the above `class_names` would throw an error).

So, here is the minimal dataset definition file:

```yaml
# Here, we are describing our dataset. There are only a couple required fields.
# The first is the class names, which gives the index-to-class-name map. In this
class_names:
  - dog
  - cat
  - bat

# We use `dataset_paths` to define the paths to folders individually.
dataset_paths:
  set1:  # this name is just for your convenience!
    image_path: /path/to/images/
    label_path: /path/to/labels/
  set2:
    image_path: /path/to/images/
    label_path: /path/to/labels/
  ...
```

## Splitting your dataset

The above definition is very simple; it defines three classes and a dataset for training. For training a model, we of course want to be able to specify more options. For example, we'd probably want to specify a validation dataset (a subset of the training data that is *not* used for updating model weights - it is used to check for overfitting during training) and a test dataset. By default, all of the data is assigned to the train partition. We can randomly split our data with the `dataset_split_fractions` key:

```yaml
...
dataset_split_fractions:
  train: 0.8
  val: 0.1
  test: 0.1
...
```

A random 80% of the image/label pairs will be assigned to train, 10% to val, 10% to test. If you would like to explicitly define the test dataset, you can use `test_dataset_paths`:

```yaml
...
dataset_paths:
  set1:
    image_path: /path/to/images1/
    label_path: /path/to/labels1/
  set2:
    image_path: /path/to/images2/
    label_path: /path/to/labels2/

test_dataset_paths:
  set3:
    image_path: /path/to/images3/
    label_path: /path/to/labels3/
```

The above definition assigns all of the data in `dataset_paths` to the train dataset, and all the data in `test_dataset_paths` to the test dataset. To randomly split some percent of the training dataset for testing, use `dataset_split_fractions` again:

```yaml
...
dataset_split_fractions:
  train: 0.8
  val: 0.2

dataset_paths:
  set1:
    image_path: /path/to/images1/
    label_path: /path/to/labels1/
  set2:
    image_path: /path/to/images2/
    label_path: /path/to/labels2/

test_dataset_paths:
  set3:
    image_path: /path/to/images3/
    label_path: /path/to/labels3/
```

Note that when `test_dataset_paths` is present, the `test` key in `dataset_split_fractions` is invalid.


## Nice features

A quick list of features that we have:

1. Both absolute and relative paths (relative to the defn. file) are both supported - e.g. the following is valid
```yaml
...
dataset_paths:
  set1:
    image_path: /path/to/images1/
    label_path: /path/to/labels1/
  set2:
    image_path: ../images2/
    label_path: ../labels2/
...
```
2. Recursive dataset definitions are supported! Say you have `/path/to/defn1.yml` and you want to incorporate those paths into a new dataset, use the `defn_path` key:
```yaml
...
dataset_paths:
  set1:
    image_path: /path/to/images1/
    label_path: /path/to/labels1/
  set2:
    defn_path: /path/to/defn1.yml
  set3:
    defn_path: ../path/to/defn2.yml  # recursive paths work here too

# if defn1.yml uses `test_dataset_paths`, to incorporate the test paths from
# defn1.yml, you need to also include it like so:
test_dataset_paths:
  set2:
    defn_path: /path/to/defn1.yml
```
> [!NOTE]
> The ability to specify another dataset definition within a dataset definition has some restrictions. The dataset definition specifcation is a graph, where the nodes are Dataset Definitions. Edges are directed, and are from the source definition to the definitions that it defines.  For practical reasons, we can't accept arbitrary graph definitions. For example, if the specification has a cycle, we will have to reject it (only trees are allowed). We'll also choose to use unique paths - that is, for any Dataset Definition in our tree, there exists only one path to it. Essentially, we're defining a [tree](https://en.wikipedia.org/wiki/Tree_(graph_theory)).
3. Thumbnail Augmentation: If you have a bunch of "thumbnails" of your classes (i.e. if you crop bounding boxes from images) sorted into folders by class names, you can augment your data with the `thumbnail_augmentation` key:
```yaml
thumbnail_augmentation:
  cats: /folder/of/thumbnails/cats
  dogs: /folder/of/thumbnails/dogs
```

[^1]: We require every label-file to have an image-file associated with it, but not the other way around. Why? Because this way, we are able to label a subset of a folder of images and go ahead and train on the labelled subset, without having to copy the labelled images to another directory.