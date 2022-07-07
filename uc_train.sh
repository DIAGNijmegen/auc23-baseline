#!/bin/bash

if [ -z "$1" ];
then
  echo "Please pass as first argument to $0 a directory that contains ./preprocessed/"
  exit
fi

if ! [ -d "$1/preprocessed" ];
then
  echo "Please make sure that $1/preprocessed/ exists"
  exit
fi

mkdir -p "$1/trained_models"

export nnUNet_raw_data_base="$1/raw"
export nnUNet_preprocessed="$1/preprocessed"
export RESULTS_FOLDER="$1/trained_models"
#export nnUNet_n_proc_DA=4

echo running "python3 -u uc_train.py ${@:2}"
python3 -u uc_train.py ${@:2}
