# Human Annotation

We will use [Label Studio](https://labelstud.io/) for human annotation. Install it from YOGO's root directory like so:

`python3 -m pip install ".[label]"`

## Time to start annotating!

1. Start Label Studio by running: `python3 run_label_studio.py`
2. In LabelStudio, click `Create Project`
  - Name your project something descriptive - e.g. the name of the Run Folder
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
  - Go to the "Data Import" tab, click "Upload Files", and import the `tasks.json` in the run folder that you are annotating. It will be somewhere in `/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM Scope/scope-parasite-data/run-sets`.
  - Click "Save"

and you are ready to annotate!
