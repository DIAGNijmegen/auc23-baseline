# Training Example 

This folder contains a universal classifier for volumetric data. 
It expects the dataset to be available 

## Table of Contents
* [Datset format](#datasetformat)
* [Building, testing and exporting your training container](#buildtestexport)
* [Implementing your own training algorithm](#implementing)

<a id="datasetformat"></a>
## Dataset format 
The format and instructions below are largely copied from the format and instructions from [nnUnet](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/), which in turn is largely copied from the format used by the Medical Segmentation Decathlon (MSD).
 
Each classification dataset is stored as a separate 'Task'. Tasks are associated with a task ID, a three digit integer (this is different from the MSD!) and a task name (which you can freely choose): Task010_PancreaticCancer has 'PancreaticCancer' as task name and the task id is 10. Tasks are stored in the `nnUnet_raw_data` folder like this:
 
```
nnUNet_raw_data/
├── Task001_COVID19Severity
├── Task002_LungNoduleMalignancy
...
└── Task010_PancreaticCancer
```
On Chansey, this folder is found here:
```/mnt/netcache/bodyct/experiments/universal_classifier_t9603/data/universal_classifier_raw_data/```

Within each task folder, the following structure is expected:
 ```
Task001_COVID19Severity/
├── ImagesTr
├── (ImagesTs)
└── dataset.json
```
###ImagesTr:
Training images. 
The image files can have any scalar type. 
Images may have multiple modalities. Imaging modalities are identified by their suffix: a four-digit integer at the end of the filename. Imaging files must therefore follow the following naming convention: case_identifier_XXXX.nii.gz. Hereby, XXXX is the modality identifier. What modalities these identifiers belong to is specified in the dataset.json file (see below).
For data from multiple modalities and/or constructed from multiple slices: For each training case, all images must have the same geometry to ensure that their pixel arrays are aligned. Also make sure that all your data are co-registered!
Example:
```
Task001_COVID19Severity/
├── ImagesTr
|   ├── covid19severity_0003_0000.nii.gz
|   ... 
|   └── covid19severity_8192_0000.nii.gz
├── ImagesTs
└── dataset.json
```
 

###ImagesTs: 
ImagesTs (optional) contains the images that belong to the test cases. The file names and data follows the same format as the data in ImagesTr.
 
dataset.json:
Any classification labels must have consecutive integer values.
 
Example:
```
{ 
  "name": "COVID-19", 
  "description": "COVID-19 RT-PCR outcome prediction and prediction of severe COVID-19, defined as death or intubation after one month, from CT. This dataset can be downloaded from the Registry of Open Data on AWS: https://registry.opendata.aws/stoic2021-training/.",
  "reference": "Assistance Publique – Hôpitaux de Paris",
  "licence":"CC-BY-NC 4.0",
  "release":"1.0 23-12-2021",
  "tensorImageSize": "3D",
  "modality": { 
    "0": "CT"
  }, 
  "labels": { 
    "COVID-19": {
      "ordinal": false,
      "values": {
        "0": "Negative RT-PCR test",
        "1": "Positive RT-PCR test"
      }
    },
    "Severe COVID-19": {
      "ordinal": false,
      "values": {
        "0": "Alive and no intubation after one month",
        "1": "Death or intubation after one month"
      }
    }
  },
  "numTraining": 2000, 
  "numTest": 0,
  "training": [
    {
      "image":"./imagesTr/covid19severity_0003_0000.nii.gz",
      "COVID-19": 1, 
      "Severe COVID-19": 0
    },
    ...
    {
      "image":"./imagesTr/covid19severity_8192_0000.nii.gz",
      "COVID-19": 1, 
      "Severe COVID-19": 0
    }
  ],
  "test": []
}
```

<a id="buildtestexport"></a>
## Building, testing, and exporting your training container 
### Building
To test if your system is set up correctly, you can run `./build.sh` (Linux) or `./build.bat` (Windows), that simply implement this command:

```docker build -t stoictrain .```

Please note that the next step (testing the container) also runs a build, so this step is not necessary if you are certain that everything is set up correctly.

<a name="testing"></a>
### Testing
To test if the docker container works as expected, make sure that your raw data is in some directory `/path/to/raw/`. 

**Note: Running the following command will delete the current contents of `../inference/artifact/`.**

You can now run:

```bash test.sh /path/to/raw/```

if you are on Linux, or:

```test.bat \path\to\raw\```

if you are on Windows.

`test.sh`/`test.bat` will build the container and train your algorithm with the public training set. It will produce trained model weights in the `../inference/artifact/` directory to be used later for generating the inference Docker container. 

If the test runs successfully, you will see you will see `Training completed.`, and your model weights will have been saved to `../inference/artifact/`.

Note: If you do not have a GPU available on your system, remove the `--gpus all` flag in `test.sh`/`test.bat` to run the test. 


<a name="exporting"></a>
### Exporting
Run `export.sh`/`export.bat` to save the Docker image to `./STOICTrain.tar.gz`. This script runs `build.sh`/`build.bat` as well as the following command:
`docker save stoictrain | gzip -c > STOICTrain.tar.gz`

Please note that it is not necessary to include the resulting `STOICTrain.tar.gz` file in your submission. The challenge organizers will build and save a new Docker image using your submission. This new Docker image will be used to train your algorithm on the full (public + private) training set.

<a id="implementing"></a>
## Implementing your own training algorithm
You can implement your own solution by altering the [main]() function in `./plan_and_preprocess.py` and the [`do_learning`](https://github.com/DIAGNijmegen/universal-classifier-t9603/blob/a9916c6a2a8c075300200e0d0c04dfffe93b0b17/training/train.py#L76) function in `./train.py`. See the documentation of this function in `./train.py` for more information.

Any additional imported packages needed for inference should be added to `./requirements.txt`, and any additional files and folders you add should be explicitly copied in the `./Dockerfile`. See `./requirements.txt` and `./Dockerfile` for examples. After implementing your own algorithm, make sure that your training codebase is ready for submission by [testing](#testing) it.

Please note that your container will not have access to the internet when executing on grand-challenge.org, so any pre-trained model weights must be present in your container image. You can test this locally using the `--network=none` option of `docker run`.

After implementing your own training Docker container, you can implement your inference container. For this, please refer to [`../inference/README.md`](https://github.com/DIAGNijmegen/universal-classifier-t9603/tree/master/inference).

### Running outside of docker
To run the code in this repo outside of docker, first add environment variables ofr your raw data, preprocessed data and results folder. See the [original nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md) for more details. An example for setting the environment variables can be found in `./run_local.sh`:
