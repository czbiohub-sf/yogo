# Data Pipeline

Having a clear, reproducable, and understandable data pipeline is absolutely necessary if we would like to maintain our sanity + scientific rigour. Here is the framework that I am using for our specific scenario (lots of "little" datasets, many various run conditions, e.t.c.).

## Initial Data Formatting and Annotation

Data from the scope is collected to `Annotated Datasets and model training` / `40x PlanAch - 160 mm TL - 405 nm` / `run-sets`

- Runs are grouped together in folders, e.g. `2022-12-14-111221-Aditi-parasites`
- These runs are in folders specifying the chip, e.g. `2022-12-13-122015__chip0562-A4_F`
- These runs have a [Zarrfile](https://zarr.readthedocs.io/en/stable/) storing the images, along with metadata, and subsample of images from the run. Example

``` console
run-sets
  2022-12-14-111221-Aditi-parasites

    2022-12-13-122015__chip0562-A4_F          # Run Folder
      2022-12-13-122015__chip0562-A4_F.zip    # Zarrfile
      metadata.csv
      sub\_sample\_images
        0.png
        ...

    2022-12-13-122015__chip0562-A4_m          # Run Folder
      2022-12-13-122015__chip0562-A4_m.zip    # Zarrfile
      metadata.csv
      sub\_sample\_images
        0.png
        ...

    ...

  2022-12-14-154742-Aditi-parasites
    ... etc  etc ...
```

YOGO and annotation requires image files, so we must convert the zarrfiles to folders of images.

Use `python3 process_zarr_to_images.py <path to run-sets>` - this will look for all Zarrfiles, and make a folder of images beside it. Each run folder would become

```console
2022-12-13-122015__chip0562-A4_F          # Run Folder
  2022-12-13-122015__chip0562-A4_F.zip    # Zarrfile
  images/                                 # folder of images from Zarrfile
    img_0000.png
    img_0001.png
    ...
  metadata.csv
  sub\_sample\_images
    0.png
    ...
```

To create bounding box labels for images, we use [Cellpose](https://www.google.com/search?client=firefox-b-d&q=Cellpose).

On Bruno (CZB's HPC), run `sbatch cellpose_label.sh <path to run-sets>`. This operation takes ~15 minutes per run.

This will create a folder of labels formatted for YOGO and for annotation. Each run folder would become

```console
2022-12-13-122015__chip0562-A4_F          # Run Folder
  2022-12-13-122015__chip0562-A4_F.zip    # Zarrfile
  images/                                 # folder of images from Zarrfile
    img_0000.png
    img_0001.png
    ...
  labels/                                 # labels for images/
    img_0000.txt
    img_0001.txt
  metadata.csv
  sub\_sample\_images
    0.png
    ...
```

At this point, `labels` should have good bounding boxes for `images`. You can verify with

`./visualize_boxes.py <path to images folder> <path to labels folder>`

(make sure you've `ssh`'d into Bruno with XTerm - e.g. `ssh account@login01.czbiohub.org -Y`)

All cells will be classified as healthy, though. To classify them further, we need human annotators. See the README for Label Studio instructions.

## Exporting from Label Studio

After annotating your images, it is time to export. If you use the "Export" button on the UI, LabelStudio will also export your unchanged images. We do not want that - we want just the labels. Therefore, we will use their API endpoint.

Click on the symbol for your account on the upper-right of the screen (for me, it is a circle with "AJ" in the center), and go to "Account & Settings". There, copy your "Authorization Token".

Note the project ID from the URL of the project. Navigate to the project from which you are exporting labels. The URL should look something like:

```
http://localhost:8080/projects/13/data?tab=9&task=2

                               ^^ "13" is the project id
```

Now, run the following, substituting in the project ID and the auth token

```console
curl -X GET "http://localhost:8080/api/projects/<project id>/export?exportType=YOLO" -H "Authorization: Token <paste the Auth. token here>" --output annotations.zip
```

Once you unzip that folder, the `labels` folder will replace the original labels folder.
