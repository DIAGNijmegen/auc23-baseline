#!/bin/bash

#export nnUNet_raw_data_base="/home/luuk/Documents/universal-classifier/dummy_data/raw"
#export nnUNet_preprocessed="/home/luuk/Documents/universal-classifier/dummy_data/preprocessed"
#export RESULTS_FOLDER="/home/luuk/Documents/universal-classifier/dummy_data/trained_models"

# redo preprocessing except for cropping
#rm -r /home/luuk/Documents/universal-classifier/dummy_data/preprocessed/
#rm /home/luuk/Documents/universal-classifier/dummy_data/raw/nnUNet_cropped_data/Task001_COVID19Severity/dataset_properties.pkl
#rm /home/luuk/Documents/universal-classifier/dummy_data/raw/nnUNet_cropped_data/Task001_COVID19Severity/intensityproperties.pkl

export nnUNet_raw_data_base="/home/lhboulogne/Documents/phd/data/dummy_data/raw"
export nnUNet_preprocessed="/home/lhboulogne/Documents/phd/data/dummy_data/preprocessed"
export RESULTS_FOLDER="/home/lhboulogne/Documents/phd/data/dummy_data/trained_models"

python3 plan_and_preprocess.py -t 1 -tf 4 -tl 4 --planner3d ClassificationExperimentPlanner3D --planner2d None
