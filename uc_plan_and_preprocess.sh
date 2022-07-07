#!/bin/bash

if [ -z "$1" ];
then
  echo "Please pass as first argument to $0 a directory that contains ./raw/"
  exit
fi

if ! [ -d "$1/raw" ];
then
  echo "Please make sure that $1/raw/ exists"
  exit
fi

mkdir -p "$1/preprocessed"

export nnUNet_raw_data_base="$1/raw"
export nnUNet_preprocessed="$1/preprocessed"
export RESULTS_FOLDER="$1/trained_models"

echo running "python3 -u uc_plan_and_preprocess.py ${@:2}"
python3 -u uc_plan_and_preprocess.py ${@:2}
