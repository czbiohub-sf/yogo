# `yogo`

This will show the basics on how to use YOGO from your command line interface. I suggest you use `yogo --help` liberally - the docs there should give you a better idea of how to use the tool! If it's unclear, [let me know](https://github.com/czbiohub-sf/yogo/issues/new) and I will change the docs here or in the `cli` tool.

I will not go through every option for each command here, but I'll point out major themes.

First, what can you do with the cli?

```console
$ yogo --help
usage: yogo [-h] {train,test,export,infer} ...

what can yogo do for you today?

positional arguments:
  {train,test,export,infer}
                        here is what you can do
    train               train a model
    test                test a model
    export              export a model
    infer               infer images using a model

options:
  -h, --help            show this help message and exit
```

## `yogo infer`

This is what you need most of the time. It lets you run YOGO on a dataset from the command line. It is essentially using the function `predict` in [infer.py](https://github.com/czbiohub-sf/yogo/blob/main/yogo/infer.py).

`yogo infer` will also automatically use a GPU if it can, and this will be way faster than running YOGO on a CPU.

```console
$ yogo infer --help
usage: yogo infer [-h] (--path-to-images PATH_TO_IMAGES | --path-to-zarr PATH_TO_ZARR) [--output-dir OUTPUT_DIR]
                  [--draw-boxes | --no-draw-boxes] [--save-preds | --no-save-preds] [--save-npy | --no-save-npy]
                  [--class-names [CLASS_NAMES ...]] [--count | --no-count] [--batch-size BATCH_SIZE] [--device [DEVICE]]
                  [--half | --no-half] [--crop-height CROP_HEIGHT] [--output-img-filetype {.png,.tif,.tiff}]
                  [--obj-thresh OBJ_THRESH] [--iou-thresh IOU_THRESH]
                  pth_path

positional arguments:
  pth_path              path to .pth file defining the model

options:
  -h, --help            show this help message and exit
  --path-to-images PATH_TO_IMAGES, --path-to-image PATH_TO_IMAGES
                        path to image or images
  --path-to-zarr PATH_TO_ZARR
                        path to zarr file
  --output-dir OUTPUT_DIR
                        path to directory for results, either --draw-boxes or --save-preds
  --draw-boxes, --no-draw-boxes
                        plot and either save (if --output-dir is set) or show each image (default: False)
  --save-preds, --no-save-preds
                        save predictions in YOGO label format - requires `--output-dir` to be set (default: False)
  --save-npy, --no-save-npy
                        Parse and save predictions in the same format as on scope - requires `--output-dir` to be set
                        (default: False)
  --class-names [CLASS_NAMES ...]
                        list of class names - will default to integers if not provided
  --count, --no-count   display the final predicted counts per-class (default: False)
  --batch-size BATCH_SIZE
                        batch size for inference (default: 64)
  --device [DEVICE]     set a device for the run - if not specified, we will try to use 'cuda', and fallback on 'cpu'
  --half, --no-half     half precision (i.e. fp16) inference (TODO compare prediction performance) (default: False)
  --crop-height CROP_HEIGHT
                        crop image verically - '-c 0.25' will crop images to (round(0.25 * height), width)
  --output-img-filetype {.png,.tif,.tiff}
                        filetype for output images (default: .png)
  --obj-thresh OBJ_THRESH
                        objectness threshold for predictions (default: 0.5)
  --iou-thresh IOU_THRESH
                        intersection over union threshold for predictions (default: 0.5)
  --min-class-confidence-threshold MIN_CLASS_CONFIDENCE_THRESHOLD
                        minimum confidence for a class to be considered - i.e. the max confidence must be greater than this
                        value (default: 0.0)
```

There are actually two required arguments here:
- `pth_path`, which is the path to the yogo `.pth` save file that you would like to run inference with.
- `--path-to-images` or `--path-to-zarr`, which is just the path to the folder of images (or single image), or the path to a zarr file of images

If you don't specify `--output-dir`, results will be displayed to you, either via the command line or from a pop-up window for `--draw-boxes`.

## `yogo train`

There are a *lot* of options here. At the most basic level, `yogo train path/to/dataset_description.yml` (docs for [`dataset_description.yml`](dataset-definition.md)) will train a model with decent default parameters. Here is `yogo train --help`:

```console
$ yogo train --help
usage: yogo train [-h] [--from-pretrained FROM_PRETRAINED]
                  [--dataset-split-override DATASET_SPLIT_OVERRIDE DATASET_SPLIT_OVERRIDE DATASET_SPLIT_OVERRIDE]
                  [-bs BATCH_SIZE] [-lr LEARNING_RATE] [--lr-decay-factor LR_DECAY_FACTOR]
                  [--label-smoothing LABEL_SMOOTHING] [-wd WEIGHT_DECAY] [--epochs EPOCHS] [--no-obj-weight NO_OBJ_WEIGHT]
                  [--iou-weight IOU_WEIGHT] [--classify-weight CLASSIFY_WEIGHT] [--healthy-weight HEALTHY_WEIGHT]
                  [--normalize-images | --no-normalize-images] [--image-hw IMAGE_HW IMAGE_HW]
                  [--rgb-images | --no-rgb-images]
                  [--model [{base_model,silu_model,double_filters,triple_filters,half_filters,quarter_filters,depth_ver_0,depth_ver_1,depth_ver_2,depth_ver_3,depth_ver_4,convnext_small}]]
                  [--half | --no-half] [--device [DEVICE]] [--note NOTE] [--name NAME] [--tags [TAGS ...]]
                  dataset_descriptor_file

positional arguments:
  dataset_descriptor_file
                        path to yml dataset descriptor file

options:
  -h, --help            show this help message and exit
  --from-pretrained FROM_PRETRAINED
                        start training from the provided pth file
  --dataset-split-override DATASET_SPLIT_OVERRIDE DATASET_SPLIT_OVERRIDE DATASET_SPLIT_OVERRIDE
                        override dataset split fractions, in 'train val test' order - e.g. '0.7 0.2 0.1' will set 70 percent
                        of all data to training, 20 percent to validation, and 10 percent to test. All of the data, including
                        paths specified in 'test_paths', will be randomly assigned to training, validation, and test.
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size for training (default: 64)
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE, --lr LEARNING_RATE
                        learning rate for training (default: 0.0003)
  --lr-decay-factor LR_DECAY_FACTOR
                        factor by which to decay lr - e.g. '2' will give a final learning rate of `lr` / 2 (default: 10)
  --label-smoothing LABEL_SMOOTHING
                        label smoothing (default: 0.01)
  -wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        weight decay for training (default: 0.05)
  --epochs EPOCHS       number of epochs to train (default: 64)
  --no-obj-weight NO_OBJ_WEIGHT
                        weight for the objectness loss when there isn't an object (default: 0.5)
  --iou-weight IOU_WEIGHT
                        weight for the iou loss (default: 5.0)
  --classify-weight CLASSIFY_WEIGHT
                        weight for the classification loss (default: 1.0)
  --healthy-weight HEALTHY_WEIGHT
                        weight for healthy class, between 0 and 1 (default: 1.0)
  --normalize-images, --no-normalize-images
                        normalize images into [0,1] - overridden if loading from pth (default: False)
  --image-hw IMAGE_HW IMAGE_HW
                        height and width of images for training (e.g. --image-hw 772 1032) (default: 772 1032)
  --rgb-images, --no-rgb-images
                        use RGB images instead of grayscale - overridden if loading from pth (defaults to grayscale)
                        (default: False)
  --model [{base_model,silu_model,double_filters,triple_filters,half_filters,quarter_filters,depth_ver_0,depth_ver_1,depth_ver_2,depth_ver_3,depth_ver_4,convnext_small}]
                        model version to use - do not use with --from-pretrained, as we use the pretrained model
  --half, --no-half     half precision (i.e. fp16) training. When true, try doubling your batch size to get best use of GPU
                        (default: False)
  --device [DEVICE]     set a device for the run - if not specified, we will try to use 'cuda', and fallback on 'cpu'
  --note NOTE           note for the run (e.g. 'run on a TI-82')
  --name NAME           name for the run (e.g. 'ti-82_run')
  --tags [TAGS ...]     tags for the run (e.g. '--tags test fine-tune')

```

There are a lot of options. Here are some recipes:

You can fine-tune YOGO by training from an existing YOGO model.
```console
$ yogo train path/to/ddf.yml --from-pretrained path/to/yogo_model_file.pth
```

Training data is logged to Weights and Biases. For W&B specifically,
```console
$ yogo train path/to/ddf.yml --note "my first training!" --tag "test training run" --name "i dont use this option frequently"
```

You can normalize input images to the network with `--normalize-images`, though I haven't found that it makes much of a difference.
```console
$ yogo train path/to/ddf.yml --normalize-images
```

You can also train the network on bounding boxes only with `--no-classify`. This is good for pre-training on data that doesn't have accurate labels.
```console
$ yogo train path/to/ddf.yml --no-classify
```

And finally, you can train with a different model architecture with `--model`. The model definitions are from [`model_defns.py`](https://github.com/czbiohub-sf/yogo/blob/main/yogo/model_defns.py). Though,I find that the default model tends to do better. Perhaps this is because I've spend a lot of time optimizing hyperparameters for it?
```console
$ yogo train path/to/ddf.yml --model model_big_simple
```

Most of the other options are for hyperparameters. They are all fairly standard.

## `yogo export`

This tool is for exporting a `.pth` file to `.onnx` and OpenVino's [IR](https://docs.openvino.ai/2023.0/openvino_ir.html). It's pretty turn-key. If you want to crop image height during inference, you can pass `--crop-height`.

```console
$ yogo export --help
usage: yogo export [-h] [--crop-height CROP_HEIGHT] [--output-filename OUTPUT_FILENAME] [--simplify | --no-simplify] input

positional arguments:
  input                 path to input pth file

options:
  -h, --help            show this help message and exit
  --crop-height CROP_HEIGHT
                        crop image verically - '-c 0.25' will crop images to (round(0.25 * height), width)
  --output-filename OUTPUT_FILENAME
                        output filename
  --simplify, --no-simplify
                        attempt to simplify the onnx model (default: True)
```

## `yogo test`

Need to test a model against some dataset? This will test the given YOGO pth file against the given dataset definition file's test set. Useful for running tests on a model checkpoint where that training run failed, e.g. due to hitting time limits.

```console
$ yogo test --help
usage: yogo test [-h] [--wandb | --no-wandb] [--wandb-resume-id WANDB_RESUME_ID] [--dump-to-disk | --no-dump-to-disk]
                 [--include-mAP | --no-include-mAP] [--include-background | --no-include-background] [--note NOTE]
                 [--tags [TAGS ...]]
                 pth_path dataset_defn_path

positional arguments:
  pth_path
  dataset_defn_path

options:
  -h, --help            show this help message and exit
  --wandb, --no-wandb   log to wandb - this will create a new run. If neither this nor --wandb-resume-id are provided, the
                        run will be saved to a new folder (default: False)
  --wandb-resume-id WANDB_RESUME_ID
                        wandb run id - this will essentially append the results to an existing run, given by this run id
  --dump-to-disk, --no-dump-to-disk
                        dump results to disk as a pkl file (default: False)
  --include-mAP, --no-include-mAP
                        calculate mAP as well - just a bit slower (default: False)
  --include-background, --no-include-background
                        include 'backround' in confusion matrix (default: False)
  --note NOTE           note for the run (e.g. 'run on a TI-82')
  --tags [TAGS ...]     tags for the run (e.g. '--tags test fine-tune')
  ```
