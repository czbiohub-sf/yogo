# Data Pipeline

Having a clear, reproducable, and understandable data pipeline is absolutely necessary if we would like to maintain sanity + scientific rigour. Here is the framework that I am using for our specific scenario (lots of "little" datasets, many various run conditions, e.t.c.).

## Initial Data Organization

Data *for each run* (so there will be many of these base folders) will have the format (roughly) of

```
base-folder
  zarrfile.zip
  metadata.czv
  sub\_sample\_images
    ...
```

`base-folder` could be fairly deep in a directory structure. Luckily, the only zip files in these folders are the `zarrfile.zip` files. So the plan will be to put the nested folders (assumed to *only* contain these `base-folder` type folders) in a specific directory (`/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM Scope/scope-parasite-data/run-sets`), and operate on that. For example,

```
run-sets/
  day-1/
    base-folder-1/
      images/
      metadata.csv
    ...
    base-folder-n/
  ...
  day-n/
    base-folder-1/
    ...
    base-folder-n/
```

## Operations on the data

We want to format this data for YOGO, optionally labelling cells w/ cellpose. So the steps are

1. format zarrfiles into folders of images (w/ same directory structure?)
2. two options:
  a. run cellpose on each of these folders optionally for labels
  b. otherwise, get labels in a different way (such as Napari)
3. label each folder

