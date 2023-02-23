# Human Annotation

We will use [Label Studio](https://labelstud.io/) for human annotation. First, install YOGO with label studio instructions [here](https://github.com/czbiohub/yogo#installation-for-annotations).

## Annotating

1. Start Label Studio by running: `python3 run_label_studio.py`. This assumes running on `OnDemand`. To run locally, mount `flexo` to your computer and run `python3 run_label_studio.py <path to run-set folder>`
2. In LabelStudio, click `Create Project`
  - Name your project the name of the run folder, or else
  - Go to "Labelling Setup" and click "Custom Template" on the left. Under the "Code" section, paste in the following XML and save
```xml
<View>
    <Image name="image" value="$image" zoom="true" zoomControl="true" />
    <Header value="RectangleLabels" />
    <RectangleLabels
        name="label"
        toName="image"
        canRotate="false"
        strokeWidth="3"
        opacity=".1"
    >
        <Label value="healthy" background="rgba(200, 255, 200, 1)" />
        <Label value="ring" background="rgba(250, 100, 150, 1)" />
        <Label value="trophozoite" background="rgba(255, 220, 200, 1)" />
        <Label value="schizont" background="rgba(255, 180, 100, 1)" />
        <Label value="gametocyte" background="rgba(255, 200, 255, 1)" />
        <Label value="wbc" background="rgba(200, 250, 255, 1)" />
        <Label value="misc" background="rgba(100, 100, 100, 1)" />
    </RectangleLabels>
</View>
```
  - Go to the "Data Import" tab, click "Upload Files", and import the `tasks.json` in the run folder that you are annotating. It will be somewhere in `scope-parasite-data/run-sets`.
  - Click "Save"

and you are ready to annotate!

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
curl -X GET "http://localhost:8080/api/projects/<project id>/export?exportType=YOLO&download_resources=false" -H "Authorization: Token <paste the Auth. token here>" --output annotations.zip
```

Send that folder to Axel. Thank you!

## Troubleshooting

### "Package Not Found" during installation

If your `pip` version is really low (e.g. version 9), try `python3 -m pip install --upgrade pip`. This could also be a sign of your python version being quite low (which occurs, e.g., with a base `conda` environment). Double check with `python3 --version`. It should be at least python 3.7.
