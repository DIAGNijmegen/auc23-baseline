# Training Example 

This repo contains preprocessing, training, and inference code for a universal 3D classifier (UC). It is heavily based on the [nnUnet](https://github.com/MIC-DKFZ/nnUNet) code base. More specifically, to make it work on SOL (the cluster at DIAG), it's based [DIAG's nnUnet fork](https://github.com/DIAGNijmegen/nnunet). It furthermore uses the [I3D model](https://github.com/hassony2/kinetics_i3d_pytorch). 

Feel free to join the discussion about this codebase in the [Universal 3D Classifier channel on Teams](https://teams.microsoft.com/l/channel/19%3a7c2d62ff6c4a4d3c9d7a8b9e2d857571%40thread.tacv2/Universal%25203D%2520classifier?groupId=97a88c45-447f-4147-9e80-5e7c013f7501&tenantId=b208fe69-471e-48c4-8d87-025e9b9a157f).

If you experience any issues with this codebase, please let us know in the [Teams channel](https://teams.microsoft.com/l/channel/19%3a7c2d62ff6c4a4d3c9d7a8b9e2d857571%40thread.tacv2/Universal%25203D%2520classifier?groupId=97a88c45-447f-4147-9e80-5e7c013f7501&tenantId=b208fe69-471e-48c4-8d87-025e9b9a157f) or open an issue on GitHub. 

## Table of Contents
* [Dataset format](#datasetformat)
* [How to run on SOL](#howtorunonsol)
  * [Preprocessing and experiment planning](#preprocessing)
  * [Training](#training)
  * [Inference](#inference)
* [Building, pushing and exporting the Docker container](#buildpushexport)
* [Running outside of docker](#running-outside-of-docker)
* [Adding to this codebase](#adding)

<a id="datasetformat"></a>
## Dataset format
The format and instructions below are largely copied from the format and instructions from [nnUnet](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/), which in turn is largely copied from the format used by the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) (MSD):

The entry point to universal classifier is the `raw_data_base/` folder. 

*DIAG specific: On Chansey, the `nnUNet_raw_data_base/` folder for for the universal classifier datasets is found here: `/mnt/netcache/bodyct/experiments/universal_classifier_t9603/data/raw/`. Please use this directory for storing your experiment data.*

Each classification dataset is stored here as a separate 'Task'. Tasks are associated with a task ID, a three digit integer (this is different from the MSD!) and a task name (which you can freely choose): Task010_PancreaticCancer has 'PancreaticCancer' as task name and the task id is 10. Tasks are stored in the `raw_data_base/nnUnet_raw_data/` folder like this:
 
```
nnUNet_raw_data/
├── Task001_COVID19Severity/
├── Task002_LungNoduleMalignancy/
...
└── Task010_PancreaticCancer/
```


Within each task folder, the following structure is expected:
 ```
Task001_COVID19Severity/
├── imagesTr/
├── (labelsTr/)
├── (imagesTs/)
├── (labelsTs/)
└── dataset.json
```
Here, the `labelsTr` , `imagesTs`, `labelsTs` directries are optional. The rest of this section describes how to structure the [`imagesTr`](#imagesTr), [`labelsTr`](#labelsTr), [`imagesTs`](#imagesTs), and [`labelsTs`](#labelsTs) directories, as well as the [`dataset.json`](dataset.json) file.

<a id="imagesTr"></a>
### imagesTr:
This directory contains your training images. The image files can have any scalar type. 

Images may have multiple modalities. Imaging modalities are identified by their suffix: a four-digit integer at the end of the filename. Imaging files must therefore follow the following naming convention: `case_identifier_XXXX.nii.gz`. Hereby, `XXXX` is the modality identifier. What modalities these identifiers belong to is specified in the `dataset.json` file (see below).

For data from multiple modalities and/or constructed from multiple slices: For each training case, all images must have the same geometry to ensure that their pixel arrays are aligned. Also make sure that all your data are co-registered!

Example for two modalities:
```
Task004_ProstateCancer/
├── imagesTr
|   ├── prostatecancer_0001_0000.nii.gz
|   ├── prostatecancer_0001_0001.nii.gz
|   ├── prostatecancer_0002_0000.nii.gz
|   ├── prostatecancer_0002_0001.nii.gz
|   ... 
|   ├── prostatecancer_0142_0000.nii.gz
|   └── prostatecancer_0142_0001.nii.gz
└── dataset.json
```

<a id="labelsTr"></a>
### labelsTs:
The universal classifier optionally receives segmentations of the region of interest (ROI) as additional input. They must contain segmentation maps that contain consecutive integers, starting with 0: 0, 1, 2, 3, ... num_labels. 0 is considered background.
```
Task012_ProstateCancer/
├── imagesTr
|   ├── prostatecancer_0001_0000.nii.gz
|   ├── prostatecancer_0001_0001.nii.gz
|   ├── prostatecancer_0002_0000.nii.gz
|   ├── prostatecancer_0002_0001.nii.gz
|   ... 
|   ├── prostatecancer_0142_0000.nii.gz
|   └── prostatecancer_0142_0001.nii.gz
├── labelsTr
|   ├── prostatecancer_0001.nii.gz
|   ├── prostatecancer_0002.nii.gz
|   ... 
|   └── prostatecancer_0142.nii.gz
└── dataset.json
```
When ROI segmentations are provided, the universal classifier:
- performs an additional preprocessing step, where it crops to the region of interest with a 5% margin around the segmentation mask.  
- feeds the ROI segmentation mask as additional input to the model.

When ROI segmentation masks are not provided, the universal classifier generates empty ROI segmentation masks (only foreground) and then uses those as input the the nnunet preprocessing pipeline. The empty masks will also be fed into the classifier as additional input.

Note that: 
- The ROI segmentation maps must have the same geometry as the data in `imagesTr` to ensure that their pixel arrays are aligned. 
- When ROI segmentation masks are provided during training, they must also be provided during inference. If they are not provided during training, they must also not be provided during inference.


<a id="imagesTs"></a>
### imagesTs: 
`imagesTs` (optional) contains the images that belong to the test cases. The file names and data follows the same format as the data in [`imagesTr`](#imagesTr).

<a id="imagesTs"></a>
### labelsTs: 
`labelsTs` (optional) contains the region of interest (ROI) segmentations that belong to the test cases. The file names and data follows the same format as the data in [`labelsTr`](#labelsTr).

 
<a id="dataset.json"></a>
### dataset.json:
An example of a `dataset.json` is given below. Please note the following: 
- The modalities listed under "modality" should correspond to the modality identifier as described in [imagesTr](#imagesTr).
- "classification_labels" is a description of the reference. Each classification label must have consecutive integer values. If multiple classification labels are given, the classifier will be trained on all of them jointly to produce multiple outputs. The "ordinal", field is currently not yet used.
- "labels" describes the region of interest (ROI) segmentation map input. This field is optional: when no ROI segmentations are provided with your dataset, please do not add the "labels" field. In this case, the universal classifier will generate empty ROI segmentation masks (only foreground) to use, and it will automatically add the "labels" field for these masks. 
- "training" describes the training dataset. Please leave out the modality identifier when specifying the paths for the training images. For each training case, each of the reference labels in "classification_labels" must also be specified. This "label" field is optional: when no segmentations are provided with your dataset, please do not add the "label" subfield to any of the training cases. In this case, the universal classifier will generate empty ROI segmentation masks (only foreground) to use, and it will add the paths to these images in the dataset.json.
- "test" is currently unused.
 
Example:
```
{
  "name": "COVID-19",
  "description": "COVID-19 RT-PCR outcome prediction and prediction of severe COVID-19, defined as death or intubation after one month, from CT. This dataset can be downloaded from the Registry of Open Data on AWS: https://registry.opendata.aws/stoic2021-training/.",
  "reference": "Assistance Publique \u2013 H\u00f4pitaux de Paris",
  "licence": "CC-BY-NC 4.0",
  "release": "1.0 23-12-2021",
  "tensorImageSize": "3D",
  "modality": {
    "0": "CT"
  },
  "classification_labels": [
    {
      "name": "COVID-19",
      "ordinal": false,
      "values": {
        "0": "Negative RT-PCR",
        "1": "Positive RT-PCR"
      }
    },
    {
      "name": "Severe COVID-19",
      "ordinal": false,
      "values": {
        "0": "Alive and no intubation after one month",
        "1": "Death or intubation after one month"
      }
    }
  ],
  "labels": {
    "0": "background",
    "1": "left upper lobe",
    "2": "left lower lobe",
    "3": "right upper lobe",
    "4": "right lower lobe",
    "5": "right middle lobe",
    "6": "lesions left upper lobe",
    "7": "lesions left lower lobe",
    "8": "lesions right upper lobe",
    "9": "lesions right lower lobe",
    "10": "lesions right middle lobe"
  },
  "numTraining": 2000,
  "numTest": 0,
  "training": [
    {
      "image":"./imagesTr/covid19severity_0003.nii.gz",
      "COVID-19": 1, 
      "Severe COVID-19": 0
      "label":"./labelsTr/covid19severity_0003.nii.gz",
    },
    ...
    {
      "image":"./imagesTr/covid19severity_8192.nii.gz",
      "COVID-19": 1, 
      "Severe COVID-19": 0
      "label":"./labelsTr/covid19severity_8192.nii.gz",
    }
  ],
  "test": []
}
```


<a id="howtorunonsol"></a>
## How to run on SOL
Running consists of three steps: [preprocessing and experiment planning](#preprocessing), [training](#training), and [inference](#inference).

<a id="preprocessing"></a>
### Preprocessing and experiment planning
To run preprocessing on SOL, you can run the following command:
```
~/c-submit --require-mem=30g --require-cpus=$NUMCPUS \
    --priority=low somesoluser luukboulogne 168 \
    doduo1.umcn.nl/universalclassifier/training:latest uc_plan_and_preprocess.sh \
    /mnt/netcache/bodyct/experiments/universal_classifier_t9603/data \
    -t $TASKID -tf $NUMCPUS -tl $NUMCPUS \
    --planner3d ClassificationExperimentPlanner3D --planner2d None
```
Notes:
- For most datasets, preprocessing should finish within 12 hours.
- For most datasets, 30G of RAM should be enough.
- `$TASKID` is the integer identifier associated with your Task name Task`$TASKID`_MYTASK. You can pass several task IDs at once.
- See `uc_plan_and_preprocess.py` for argument options and descriptions.

<a id="training"></a>
### Training
To run training on SOL, you can run the following command:
```
~/c-submit --require-mem=30g --require-cpus=$NUMCPUS \
    --priority=low somesoluser luukboulogne 168 \
    doduo1.umcn.nl/universalclassifier/training:latest train_on_sol.sh \
    /mnt/netcache/bodyct/experiments/universal_classifier_t9603/data \
    $TASKID $FOLD
```
for `$FOLD` in [0, 1, 2, 3, 4].

Notes:
- Please make sure to run `uc_train_on_sol.sh` and not `uc_train.sh`! `uc_train.sh` does not first copy the training data to the node on which you are running, so training will be a lot slower. It can slow down I/O for other SOL users as well.
- See `train.py` for argument options and descriptions.

<a id="inference"></a>
### Inference
To run inference on SOL, you can run the following command:

```
~/c-submit --require-mem=30g --require-cpus=$NUMCPUS \
    --priority=low somesoluser luukboulogne 168 \
    doduo1.umcn.nl/universalclassifier/training:latest predict.sh \
    -i /path/to/input_images \
    -s /path/to/input_roi_segmentations \
    -o /path/to/output \
```
Notes:
- Please remove the `-s /path/to/input_roi_segmentations` if you did not provide input segmentations during training.
- `/path/to/input_images` should have a flat folder structure (no subdirectories). The filenames in it (and optionally in `/path/to/input_roi_segmentations`) should follow the same format described as [imagesTr](#imagesTr) (and optionally [labelsTr](#labelsTr)).


<a id="buildpushexport"></a>
## Building, pushing, and exporting your training container 
*DIAG specific: You don't have to build push or export the container to run it on SOL. Instead, you can immediately [run the already pushed version on SOL](#howtorunonsol)*

<a name="building"></a>
### Building
To test if your system is set up correctly, you can run

`./build.sh $VERSION`.

This build command is already contained in the scripts for [pushing](#pushing) and [exporting](#exporting).

<a name="pushing"></a>
### *Pushing (DIAG specific)*
*Run `./push.sh $VERSION` to build the Docker image and push it to doduo.*

<a name="exporting"></a>
### Exporting
Run `export.sh`/`export.bat` to save the Docker image to `./universalclassifier_training_v$VERSION.tar.gz`. This script also includes `build.sh $VERSION`.


<a name="running-outside-of-docker"></a>
## Running outside of docker
To run the code in this repo outside of docker, you can call  `./uc_plan_and_preprocess.py`, `./uc_train.py`, and `./uc_predict.py` directly. These implement the universalclassifier version of the similarly named nnunet commands. 

Before doing so, please make sure to first add environment variables for your raw data, preprocessed data and results folder. Examples for setting the environment variables can be found in `./uc_plan_and_preprocess.sh`, `./uc_train.sh`, and `./uc_predict.sh`.

 See the [original nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md) for more details about the environment variables.
 
<a name="adding"></a>
## Adding to the codebase
Feel free to open an issue on GitHub or start a pull request. Also, feel free to join our disussions about this codebase in the [Universal 3D Classifier channel on Teams](https://teams.microsoft.com/l/channel/19%3a7c2d62ff6c4a4d3c9d7a8b9e2d857571%40thread.tacv2/Universal%25203D%2520classifier?groupId=97a88c45-447f-4147-9e80-5e7c013f7501&tenantId=b208fe69-471e-48c4-8d87-025e9b9a157f).