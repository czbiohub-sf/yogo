# `yogo`

This will show the basics on how to use YOGO from your command line interface. I suggest you use `yogo --help` liberally - the docs there should give you a better idea of how to use the tool! If it's unclear, [let me know](https://github.com/czbiohub-sf/yogo/issues/new) and I will change the docs here or in the `cli` tool.

I will not go through every option for each command here, but I'll point out major themes.

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

There are three options, as you can see above.

## `yogo infer`

This is what you need most of the time. It lets you infer YOGO on a dataset from the command line. For specifics about this functionality, look at [infer.py](https://github.com/czbiohub-sf/yogo/blob/main/yogo/infer.py).

```console
$ yogo infer --help
usage: yogo infer [-h] [--output-dir OUTPUT_DIR] [--draw-boxes | --no-draw-boxes]
                  [--save-preds | --no-save-preds] [--count | --no-count] [--batch-size BATCH_SIZE]
                  [--crop-height CROP_HEIGHT] [--output-img-filetype {.png,.tif,.tiff}]
                  [--obj-thresh OBJ_THRESH] [--iou-thresh IOU_THRESH]
                  (--path-to-images PATH_TO_IMAGES | --path-to-zarr PATH_TO_ZARR)
                  pth_path

positional arguments:
  pth_path              path to .pth file defining the model

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        path to directory for results, either --draw-boxes or --save-preds
  --draw-boxes, --no-draw-boxes
                        plot and either save (if --output-dir is set) or show each image (default: False)
  --save-preds, --no-save-preds
                        save predictions in YOGO label format - requires `--output-dir` to be set
                        (default: False)
  --count, --no-count   display the final predicted counts per-class (default: False)
  --batch-size BATCH_SIZE
                        batch size for inference (default 16)
  --crop-height CROP_HEIGHT
                        crop image verically - '-c 0.25' will crop images to (round(0.25 * height),
                        width)
  --output-img-filetype {.png,.tif,.tiff}
                        filetype for output images (default .png)
  --obj-thresh OBJ_THRESH
                        objectness threshold for predictions (default 0.5)
  --iou-thresh IOU_THRESH
                        intersection over union threshold for predictions (default 0.5)
  --path-to-images PATH_TO_IMAGES
                        path to image or images
  --path-to-zarr PATH_TO_ZARR
                        path to zarr file
```

There are actually two required arguments here:
- `pth_path`, which is the path to the yogo `.pth` save file that you would like to run inference with.
- `--path-to-images` or `--path-to-zarr`, which is just the path to the folder of images (or single image), or the path to a zarr file of images

If you don't specify `--output-dir`, results will be displayed to you, either via the command line or from a pop-up window for `--draw-boxes`.
