# Recipes

There are three components to running YOGO:
- Creating the model
- Loading images
- Processing the output

## Creating the model

```python3
>>> from yogo.model import YOGO

# look in `yogo_models` for the newest best model; older models will be in `yogo_models/older-models`
>>> model_dir_path = "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/yogo_models/honest-sweep-51/best.pth"

>>> Y, cfg = YOGO.from_pth(
...     model_dir_path,
...     inference=True  # set inference to True to do inference, False to do training
... )

# `Y` is the YOGO model, and cfg is a configuration for the model.
>>> print(cfg)
{'step': 107250, 'normalize_images': False}
# you can probably ignore `step` which is just the number of steps that the model was trained for.
# if `normalize_images` is True, you must rescale the input images into the range [0,1].
```

## Loading images

There are many ways to load images, and YOGO may have some tools to make that easy. You will have to consider the performance regime under which you are running YOGO to make the right choice. Here is what I would do for some cases:

#### Running a single image from disk

This is the simplest example.

```python3
>>> from torchvision.io import read_image, ImageReadMode

# image should be (772, 1032) grayscale images
>>> img = read_image("path/to/img.png", ImageReadMode.GRAY)
>>> img = img / 255 if cfg["normalize_images"] else img
>>> img.shape
torch.Size([772, 1032])

# we need to add a "channel" dimension to the image
>>> img = img.unsqueeze(0)
>>> img.shape
torch.Size([1, 772, 1032])

>>> out = Y(img)
```

Note that you can also use `PIL`, or `opencv`, or whatever other package that you want. You just have to make sure that the image is a [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#tensor-class-reference) before passing it to YOGO. You can also speed up computation and reduce memory usage by disabling [autograd](https://pytorch.org/docs/stable/notes/autograd.html) (worth a read), as long as you are not doing any training!

```python3
# either disable grad locally
>>> with torch.no_grad():
...     out = Y(img)

# or disable it globally
>>> torch.set_grad_enabled(False)
```

#### Running many images from disk

When running many images from disk, most likely you want to run them quickly. In this case, you want to load / preprocess images in the background and collate them into a batch before you feed it to YOGO. I suggest using a PyTorch [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) and [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). Implementing a dataset for your needs is very easy, and the PyTorch dataloader is very easy to tune to your needs. You may also be able to use one that I've already written if your use case is one of these:

- [Loading images and image paths](https://github.com/czbiohub-sf/yogo/blob/main/yogo/infer.py)

`ImagePathDataset` will load and return the image and path to the image. The image path can be useful, so we include it here. Of course, you can also just ignore it if you only want images. Also note the creation of the `DataLoader`. It's very short, and tuning it to your system can speed up inference by a huge amount. I highly suggest you read the `DataLoader` documentation.

- [Loading images and labels](https://github.com/czbiohub-sf/yogo/blob/main/yogo/data/dataset.py)

`ObjectDetectionDataset` will load and return an image and corresponding label, given a folder of images and the corresponding folder of labels. It is useful if you are doing some training, or maybe if you want to visualize bounding boxes from labelled data.

Taking the dataset and dataloader from `infer.py`, create your dataset and dataloader, and iterate through images:

```python3
>>> dataset = ImagePathDataset(
...   path_to_images,
...   image_transforms=image_transforms,
...   normalize_images=normalize_images,
)

>>> def collate_fn(
...     batch: List[Tuple[torch.Tensor, str]]
... ) -> Tuple[torch.Tensor, Tuple[str]]:
...     """
...     This function takes a list of (image, path) pairs and returns a batch.
...     Used by the DataLoader.
...     """
...     images, fnames = zip(*batch)
...     return torch.stack(images), cast(Tuple[str], fnames)

>>> dataloader = DataLoader(
...     dataset,
...     batch_size=16  # how big of a batch do you want? Typically you want this as big
...     collate_fn=collate_fn,
...     num_workers=min(torch.multiprocessing.cpu_count(), 32)  # number of workers for loading images. much faster for num > 0, but it needs to be tuned to your hardware
... )

>>> torch.set_grad_enabled(False)

# GPUs makes inference *way* faster. I am not going to get into many details here,
# but I'll cover 95% of the use cases w/ these examples
>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # it is good to select device-agnostic code like this
>>> Y.to(device)  # put YOGO on the GPU! many errors will occur from forgetting to do this

>>> for img_batch, paths in dataloader:
...     img_batch = img_batch.to(device)
...     out = Y(img_batch)
```

## Processing YOGO output

When you run YOGO, you'll get a 4-d tensor back:

```python3
>>> out = Y(img).shape
>>> out
torch.Size([1, 12, 97, 129])

# out[i, :, :, :] is the 'batch', so if you give YOGO a batch of images, the slice out[3, :, :, :] will give the YOGO result for the 3rd image
# out[:, j, :, :] is the 'prediction' dimension; out[:, :4, :, :] is [xc, yc, w, h], out[:, 4, :, :] is objectness, and out[:, 5:, :, :] is the class prediction
# out[:, :, k, l] is the grid dimension
```

See [docs/README.md](https://github.com/czbiohub-sf/yogo/blob/main/docs/README.md) for a little bit about the output tensor, or the latter [slides](https://docs.google.com/presentation/d/1p9k6aFVJeEl7MH0iic_kju4Ub_uUJPdb6UqJvk63rAM/edit?usp=sharing) for a presentation I gave on YOGO.

Note that this output is entirely unprocessed. If you want to filter for objectness or area, apply Non-Maximal Supression (NMS), and format the tensor into a simpler format, use [`format_preds`](https://github.com/czbiohub-sf/yogo/blob/c4d4388983968bbef5decca00aad9aecdb33362b/yogo/utils/utils.py#L132).

This will apply objectness thresholding (filtering out predictions were YOGO doesn't think there is a cell), area thresholding (filtering out small bboxes), NMS (removes double bounding boxes), and will also convert the bounding boxes to `xyxy` (top left and bottom right) format.
