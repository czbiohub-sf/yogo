# Human Annotation

We will use [Label Studio](https://labelstud.io/) for human annotation. Install it via instructions from the website:

`pip3 install -U label-studio`

And install [Label Studio Converter](https://github.com/heartexlabs/label-studio-converter) by cloning the repo and building from source.

```console
$ git clone https://github.com/heartexlabs/label-studio-converter.git
$ cd label-studio-converter
$ pip3 install .
```

## Run Label Studio

1. Start image server by running: `./serve_local_files.sh "<path to IMAGE dir in run folder>"`, using the modified version in this repo, which was adapted from the original in `label-studio`.
  - **NOTE** Make sure there aren't back-slashes in `"<path to IMAGE dir in run folder>"` - an example of a good path would be `"/im/a/path/with a space/but/its/ok"`. Since we have quotes, we don't escape the spaces.
2. Convert labels to Label Studio format: `label-studio-converter import yolo -i "<path to RUN folder>" -o tasks.json --image-ext ".png" --image-root-url http\://localhost\:8081/ --out-type predictions` (this can take some time). In the end, it should create a "tasks.json" file. Then run `sed -i '' 's/%3A/:/g' tasks.json` on Mac, and `sed -i 's/%3A/:/g' tasks.json` on Linux
3. Start Label Studio: `label-studio start`
4. Click `Create Project`
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
        <Label value="ring" background="rgba(250, 250, 150, 1)" />
        <Label value="trophozoite" background="rgba(255, 220, 200, 1)" />
        <Label value="schizont" background="rgba(255, 180, 100, 1)" />
        <Label value="gametocyte" background="rgba(255, 200, 255, 1)" />
        <Label value="wbc" background="rgba(200, 250, 255, 1)" />
        <Label value="misc" background="rgba(100, 100, 100, 1)" />
    </RectangleLabels>
</View>
```
  - Go to the "Data Import" tab, click "Upload Files", and import `tasks.json`
  - Click "Save"

and you are ready to annotate!

## Exporting

After annotating your images, it is time to export. If you use the "Export" button on the UI, LabelStudio will also export your unchanged images. We do not want that - we want just the labels. Therefore, we will use their API endpoint.

Click on the symbol for your account on the upper-right of the screen (for me, it is a circle with "AJ" in the center), and go to "Account & Settings". There, copy your "Authorization Token".

Note the project ID from the URL of the project. Navigate to the project from which you are exporting labels. The URL should look something like:

```
http://localhost:8080/projects/13/data?tab=9&task=2

                               ^^ "13" is the project id
```


Now, run the following, substituting in the project ID and the auth token

`curl -X GET "http://localhost:8080/api/projects/<project id>/export?exportType=YOLO" -H "Authorization: Token <paste the Auth. token here>" --output annotations.zip`

Once you unzip that folder, the `labels` folder will replace the original labels folder.
