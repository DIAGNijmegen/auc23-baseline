#!/bin/bash

if [ -z "$1" ];
then
  echo "Please pass as first argument to $0 a directory that contains ./trained_models/"
  exit
fi

if ! [ -d "$1/trained_models" ];
then
  echo "Please make sure that $1/trained_models/ exists"
  exit
fi

export nnUNet_raw_data_base="$1/raw"
export nnUNet_preprocessed="$1/preprocessed"
export RESULTS_FOLDER="$1/trained_models"

echo running "python3 -u uc_predict.py ${@:2}"
python3 -u uc_predict.py ${@:2}
