# `yogo`

This will show the basics on how to use YOGO from your command line interface. I suggest you use `yogo --help` liberally - the docs there should give you a better idea of how to use the tool! If it's unclear, [let me know](https://github.com/czbiohub-sf/yogo/issues/new) and I will change the docs here or in the `cli` tool.

I will not go through every option for each command here, but I'll point out major themes.

First, what can you do with the cli?

```console
$ yogo --help
usage: yogo [-h] {train,export,infer} ...

what can yogo do for you today?

positional arguments:
  {train,export,infer}  here is what you can do
    train               train a model
    export              export a model
    infer               infer images using a model

optional arguments:
  -h, --help            show this help message and exit
```

## `yogo infer`

This is what you need most of the time. It lets you run YOGO on a dataset from the command line. It is essentially using the function `predict` in [infer.py](https://github.com/czbiohub-sf/yogo/blob/main/yogo/infer.py).

`yogo infer` will also automatically use a GPU if it can, and this will be way faster than running YOGO on a CPU.

```console
$ yogo infer --help
usage: yogo infer [-h] [--output-dir OUTPUT_DIR] [--draw-boxes | --no-draw-boxes] [--save-preds | --no-save-preds]
                  [--save-npy | --no-save-npy] [--count | --no-count] [--batch-size BATCH_SIZE] [--device [DEVICE]]
                  [--crop-height CROP_HEIGHT] [--output-img-filetype {.png,.tif,.tiff}] [--obj-thresh OBJ_THRESH]
                  [--iou-thresh IOU_THRESH] [--min-class-confidence-threshold MIN_CLASS_CONFIDENCE_THRESHOLD]
                  [--heatmap-mask-path HEATMAP_MASK_PATH] (--path-to-images PATH_TO_IMAGES | --path-to-zarr PATH_TO_ZARR)
                  pth_path

positional arguments:
  pth_path              path to .pth file defining the model

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        path to directory for results, either --draw-boxes or --save-preds
  --draw-boxes, --no-draw-boxes
                        plot and either save (if --output-dir is set) or show each image (default: False)
  --save-preds, --no-save-preds
                        save predictions in YOGO label format - requires `--output-dir` to be set (default: False)
  --save-npy, --no-save-npy
                        Parse and save predictions in the same format as on scope - requires `--output-dir` to be set (default:
                        False)
  --count, --no-count   display the final predicted counts per-class (default: False)
  --batch-size BATCH_SIZE
                        batch size for inference (default: 64)
  --device [DEVICE]     set a device for the run - if not specified, we will try to use 'cuda', and fallback on 'cpu'
  --crop-height CROP_HEIGHT
                        crop image verically - '-c 0.25' will crop images to (round(0.25 * height), width)
  --output-img-filetype {.png,.tif,.tiff}
                        filetype for output images (default: .png)
  --obj-thresh OBJ_THRESH
                        objectness threshold for predictions (default: 0.5)
  --iou-thresh IOU_THRESH
                        intersection over union threshold for predictions (default: 0.5)
  --min-class-confidence-threshold MIN_CLASS_CONFIDENCE_THRESHOLD
                        minimum confidence for a class to be considered - i.e. the max confidence must be greater than this value
                        (default: 0.0)
  --heatmap-mask-path HEATMAP_MASK_PATH
                        path to heatmap mask for the run (default: None)
  --path-to-images PATH_TO_IMAGES
                        path to image or images
  --path-to-zarr PATH_TO_ZARR
                        path to zarr file
```

There are actually two required arguments here:
- `pth_path`, which is the path to the yogo `.pth` save file that you would like to run inference with.
- `--path-to-images` or `--path-to-zarr`, which is just the path to the folder of images (or single image), or the path to a zarr file of images

If you don't specify `--output-dir`, results will be displayed to you, either via the command line or from a pop-up window for `--draw-boxes`.

## `yogo train`

There are a *lot* of options here. At the most basic level, `yogo train path/to/dataset_description.yml` (docs for [`dataset_description.yml`](dataset-definition.md)) will train a model with decent default parameters. Here is `yogo train --help`:

```console
$ yogo train --help
usage: yogo train [-h] [--from-pretrained FROM_PRETRAINED] [-bs BATCH_SIZE] [-lr LEARNING_RATE] [--lr-decay-factor LR_DECAY_FACTOR]
                  [--label-smoothing LABEL_SMOOTHING] [-wd WEIGHT_DECAY] [--epochs EPOCHS] [--no-obj-weight NO_OBJ_WEIGHT]
                  [--iou-weight IOU_WEIGHT] [--classify-weight CLASSIFY_WEIGHT] [--healthy-weight HEALTHY_WEIGHT]
                  [--no-classify | --no-no-classify] [--normalize-images | --no-normalize-images]
                  [--image-shape IMAGE_SHAPE IMAGE_SHAPE]
                  [--model [{base_model,smaller_funkier,even_smaller_funkier,model_no_dropout,model_smaller_SxSy,model_big_simple,model_big_residual,model_big_normalized,model_big_heavy_normalized,convnext_small}]]
                  [--half | --no-half] [--device [DEVICE]] [--note NOTE] [--name NAME] [--tag TAG]
                  dataset_descriptor_file

positional arguments:
  dataset_descriptor_file
                        path to yml dataset descriptor file

options:
  -h, --help            show this help message and exit
  --from-pretrained FROM_PRETRAINED
                        start training from the provided pth file
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
  --no-classify, --no-no-classify
                        turn off classification loss - good only for pretraining just a cell detector (default: False)
  --normalize-images, --no-normalize-images
                        normalize images into [0,1] (default: False)
  --image-shape IMAGE_SHAPE IMAGE_SHAPE
                        size of images for training (e.g. --image-shape 772 1032) (default: 772 1032)
  --model [{base_model,smaller_funkier,even_smaller_funkier,model_no_dropout,model_smaller_SxSy,model_big_simple,model_big_residual,model_big_normalized,model_big_heavy_normalized,convnext_small}]
                        model version to use - do not use with --from-pretrained, as we use the pretrained model
  --half, --no-half     half precision (i.e. fp16) training. When true, try doubling your batch size to get best use of GPU.
                        (default: False) (default: False)
  --device [DEVICE]     set a device for the run - if not specified, we will try to use 'cuda', and fallback on 'cpu'
  --note NOTE           note for the run (e.g. 'run on a TI-82')
  --name NAME           name for the run (e.g. 'ti-82_run')
  --tag TAG             tag for the run (e.g. 'test')
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
