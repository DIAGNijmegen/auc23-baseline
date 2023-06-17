# AUC23 baseline

This repo contains preprocessing, training, and inference code for a universal 3D medical image classifier. It is heavily based on the [nnUnet](https://github.com/MIC-DKFZ/nnUNet) code base. More specifically, it's based on [the nnUnet fork of the Diagnostic Image Analysis Group](https://github.com/DIAGNijmegen/nnunet). It furthermore uses the [I3D model](https://github.com/hassony2/kinetics_i3d_pytorch).

If you experience any issues with this codebase, please open an issue on GitHub. 

This codebase can be used as a template for submitting new universal 3D medical image classification solutions to https://auc23.grand-challenge.org/. This readme contains a [tuturial for submitting to AUC23](#submitting). 

## Table of Contents
* [Dataset format](#datasetformat)
* [Running the codebase](#running-the-codebase)
* [Submitting to AUC23](#submitting)

<a id="datasetformat"></a>
## Dataset format
The format and instructions below are largely copied from the format and instructions from [nnUnet](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/), which in turn is largely copied from the format used by the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) (MSD):

The entry point to universal classifier is your `/path/to/data/raw` folder.

Each classification dataset is stored here as a separate 'Task'. Tasks are associated with a task ID, a three digit integer (this is different from the MSD!) and a task name (which you can freely choose): Task010_PancreaticCancer has 'PancreaticCancer' as task name and the task id is 10. Tasks are stored in the `raw/nnUnet_raw_data/` folder like this:
 
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
### labelsTr:
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
- **The modality identifier is used in the actual image filenames ([see imagesTr](#imagesTr)) , but not in the filenames in `dataset.json`.**  
- The modalities listed under "modality" should correspond to the modality identifiers as described in [imagesTr](#imagesTr).
- "classification_labels" is a description of the reference. Each classification label must have consecutive integer values. If multiple classification labels are given, the classifier will be trained on all of them jointly to produce multiple outputs. The "ordinal" field is currently not yet used.
- "labels" describes the region of interest (ROI) segmentation map input. This field is optional: when no ROI segmentations are provided with your dataset, please do not add the "labels" field. In this case, the universal classifier will generate empty ROI segmentation masks (only foreground) to use, and it will automatically add the "labels" field for these masks. 
- "training" describes the training dataset. **Please leave out the modality identifier when specifying the paths for the training images.** For each training case, each of the reference labels in "classification_labels" must also be specified. This "label" field is optional: when no segmentations are provided with your dataset, please do not add the "label" subfield to any of the training cases. In this case, the universal classifier will generate empty ROI segmentation masks (only foreground) to use, and it will add the paths to these images in the dataset.json.
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
      "Severe COVID-19": 0,
      "label":"./labelsTr/covid19severity_0003.nii.gz",
    },
    ...
    {
      "image":"./imagesTr/covid19severity_8192.nii.gz",
      "COVID-19": 1, 
      "Severe COVID-19": 0,
      "label":"./labelsTr/covid19severity_8192.nii.gz",
    }
  ],
  "test": []
}
```




<a name="running-the-codebase"></a>
## Running preprocessing, training, and inference 
Running consists of three steps: [preprocessing and experiment planning](#preprocessing), [training](#training), and [inference](#inference). The root folder in which this codebase will operate will be some `/path/to/data/` directory of your choice. Make sure your dataset(s) are stored in `/path/to/data/raw`, as explained in [Dataset format](#datasetformat). Also, make sure the directories `/path/to/data/preprocessed/`, and `/path/to/data/trained_models/` exist.

<a id="preprocessing"></a>
### Preprocessing and experiment planning
To run preprocessing, run the following command:
```
bash uc_plan_and_preprocess.sh \
    /path/to/data \
    -t $TASKID -tf $NUMCPUS -tl $NUMCPUS
```
Notes:
- For most datasets used for nnUnet, 60G of RAM should be enough.
- `$TASKID` is the integer identifier associated with your Task name Task`$TASKID`_MYTASK. You can pass several task IDs at once.
- See `uc_plan_and_preprocess.py` for argument options and descriptions.
- Don't use too many CPUs ($NUMCPUS), or you will run out or RAM. It's recommended to use 4 CPUs.

<a id="training"></a>
### Training
To run training, run the following command:
```
bash uc_train.sh \
    /path/to/data \
    3d_fullres ClassifierTrainer $TASKID $FOLD
```
for `$FOLD` in [0, 1, 2, 3, 4].

Notes:
- See `uc_train.py` for argument options and descriptions.
- If you already trained the fold, but the validation step failed, run the validation step using the --validation_only flag. 

<a id="inference"></a>
### Inference
To run inference, run the following command:

```
bash uc_predict.sh \
    /path/to/data \
    -i /path/to/input_images \
    -s /path/to/input_roi_segmentations \
    -o /path/to/output \
    -t $TASKID -f 0 1 2 3 4
```
Notes:
- ! Make sure to remove the `-s /path/to/input_roi_segmentations` if you did not provide input segmentations during training.
- `/path/to/input_images` should have a flat folder structure (no subdirectories). The filenames in it (and optionally in `/path/to/input_roi_segmentations`) should follow the same format described as [imagesTr](#imagesTr) (and optionally [labelsTr](#labelsTr)).
- See `uc_predict.py` for argument options and descriptions



### Running without bash
To run the code in this repo without bash, you can call  `./uc_plan_and_preprocess.py`, `./uc_train.py`, and `./uc_predict.py` directly. These implement the universalclassifier version of the similarly named nnunet commands. 

Before doing so, please make sure to first add environment variables for your raw data, preprocessed data and results folder. Examples for setting the environment variables can be found in `./uc_plan_and_preprocess.sh`, `./uc_train.sh`, and `./uc_predict.sh`.

See the [original nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md) for more details about the environment variables.


<a name="submitting><\a>
## Submitting your Algorithm to a Leaderboard for AUC23
Please read this tutorial before starting to develop your codebase for submission to AUC23.

You can use this codebase as a starting point for submitting your own universal 3D medical image classification solution as an Algorithm on [auc23.grand-challenge.org](auc23.grand-challenge.org). Although you can only submit trained models to the AUC23 challenge, please make sure that it is possible not only to perform inference, but also to perform preprocessing and training as detailed in [Running this codebase without Docker](#rundocker). Please note that, once submitted to grand-challenge.org, your Algorithm will not have access to the internet. Please make sure that e.g. any pre-trained weights do not need to be downloaded at runtime, but are instead pre-downloaded and present in your code base.

After you have trained your solution for one of the tasks of AUC23, you can make a submission to the corresponding task-specific Leaderboard on [auc23.grand-challenge.org](auc23.grand-challenge.org). To make a submission, follow the link below corresponding to the Leaderboard that you want to submit to:

- https://github.com/DIAGNijmegen/auc23-t2-algorithm-container-brain-mri-glioma
- https://github.com/DIAGNijmegen/auc23-t2-algorithm-container-prostate-mri-clinically-significant-cancer
- https://github.com/DIAGNijmegen/auc23-t2-algorithm-container-lung-nodule-ct-false-positive-reduction
- https://github.com/DIAGNijmegen/auc23-t2-algorithm-container-rib-ct-fracture
- https://github.com/DIAGNijmegen/auc23-t2-algorithm-container-breast-mri-molecular-cancer-type
- https://github.com/DIAGNijmegen/auc23-t2-algorithm-container-retina-oct-glaucoma
- https://github.com/DIAGNijmegen/auc23-t2-algorithm-container-lung-ct-covid-19

These urls each link to a task-specific GitHub repository containing a tutorial that describes how to create an Algorithm submission on grand-challenge to the corresponding Leaderboard. Note that the tutorials in these task-specific GitHub repositories are all identical. The GitHub repositories nevertheless differ from each other: They contain different configuration files to account for the different input and output types of each task, and they each contain trained baseline solution weights for different tasks. 

As a prerequisite, these tutorials require you to have created a pip installable `.whl` file from your solution codebase. We recommend to first create a `.whl` file for the AUC23 baseline that this repository contains to get familiar with this. The rest of this section explains how to create this `.whl` file.


###  Creating an installable `.whl` your solution codebase 
To submit your Algorithm, it is required that you package your codebase in a `.whl` file. We recommend using the `poetry` tool to create your `.whl` file. This section shows an example of how to create a `.whl` file for the baseline codebase implemented in this template repository. 

#### Prerequisites
1. Make sure that the root of your codebase contains a directory called `./universalclassifier/` that contains all source code for running training and inference with your Algorithm. This source code should implement a function `predict()` that can be imported from the root of your codebase by running `from universalclassifier import predict`. After you have trained a model, this function will be used to run that model on input images when submitting your Algorithm on [auc23.grand-challenge.org](auc23.grand-challenge.org).
  
    The `predict()` function should expect the following arguments: 
      * `artifact_path: str`: Path to a directory containing the files that describe your trained model.
      * `ordered_image_files: List[str]`: Ordered list of paths to input image files. This list will follow the same order as the modalities in the `"modality"` field of the `dataset.json` of the corresponding training dataset. 
      * `roi_segmentation_file: str = None`: Path to the region of interest segmentation file. `None` will be passed if no ROI segmentation file is provided for the task at hand.
2. Compile a `requirements.txt` file listing all required python packages for your code in  `./universalclassifier/` and place this in the root of your repository.
3. Make sure a `README.md` file exists in the root of your repository.
4. Make sure to have `poetry` installed (or another tool of your choice to build a `.whl` file from your codebase). You can install `poetry` by following these instructions https://python-poetry.org/docs/#installation.

#### Creating the `.whl` file
1. Go to the root of your codebase and run 
    ```
    poetry init
    ```
    This will ask a few questions to create a basic `pyproject.toml` file in the current directory.
    Fill in the following:

   - `Package name [directoryname]`: `universalclassifier`
   - `Version [0.1.0]`: Any version number
   - `Description []`: Any description
   - `Author`: Any author
   - `Licence`: The permissive open source licence for your codebase
   - `Compatible Python versions`: e.g. `^3.8` for the baseline
   - `Would you like to define your main dependencies interactively? (yes/no) [yes]`: no
   - `Would you like to define your development dependencies interactively? (yes/no) [yes]`: no
   - `Do you confirm generation? (yes/no) [yes]`: yes

    Your directory should now have the following structure:
    ```
    ├── universalclassifier/
    │   ├── # subdirectories and files for the universalclassifier package
    │   └── 
    ├── pyproject.toml
    ├── README.md
    └── requirements.txt
   ```

2.  Add requirements by running
    ```
    cat requirements.txt | xargs poetry add
    ```
    This command reads the package requirements from the `requirements.txt` file and adds them to the `poetry` configuration.
3.  Run 
    ```
    poetry build
    ```
    The resulting `.whl` file can be found in the `./dist/` directory created by poetry. You can now use this `.whl` file in your submission. 



If you later update your codebase, you can update the release number in `pyproject.toml` accordingly and then generate a new `.whl` file for your universal classifier with the following steps:

1. Open the `pyproject.toml` file and update the version number under `[tool.poetry]`:
    ````
    [tool.poetry]
    name = "universalclassifier"
    version = "X.Y.Z"
    ...
    ````
    Replace `X.Y.Z` with the new version number.

2. Add or update any dependencies in the `pyproject.toml` file, if needed. You can also add dependencies using the command:
    ```
    poetry add <package-name>
    ```
    Replace `<package-name>` with the name of the package you want to add.

3. Run the following command to build the new `.whl` file:
    ```
    poetry build
    ```
   The updated `.whl` file will be generated in the `./dist/` directory.
