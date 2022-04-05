#!/bin/bash

#export nnUNet_raw_data_base="/home/luuk/Documents/universal-classifier/dummy_data/raw"
#export nnUNet_preprocessed="/home/luuk/Documents/universal-classifier/dummy_data/preprocessed"
#export RESULTS_FOLDER="/home/luuk/Documents/universal-classifier/dummy_data/trained_models"

export nnUNet_raw_data_base="/home/lhboulogne/Documents/phd/data/dummy_data/raw"
export nnUNet_preprocessed="/home/lhboulogne/Documents/phd/data/dummy_data/preprocessed"
export RESULTS_FOLDER="/home/lhboulogne/Documents/phd/data/dummy_data/trained_models"

python3 main.py -t 1 -tf 2 -tl 2 --planner3d ClassificationExperimentPlanner3D #--verify_dataset_integrity