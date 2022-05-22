#!/bin/bash

if [ -z "$1" ];
do
  echo "Please pass as first argument to $0 a directory that contains ./preprocessed/"
  exit
fi

if ! [ -d "$1/preprocessed"];
do
  echo "Please make sure that $1/preprocessed/ exists"
  exit
fi

python3 copy_preprocessed.py $4 $1/preprocessed /preprocessed  # $4 is the task name or task ID
mkdir -p "$1/trained_models"

#export nnUNet_raw_data_base="$1/raw"
export nnUNet_preprocessed="/preprocessed"
export RESULTS_FOLDER="$1/trained_models"

python3 train.py ${@:2}
