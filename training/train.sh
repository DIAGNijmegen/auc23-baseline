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

mkdir -p "$1/trained_models"

#export nnUNet_raw_data_base="$1/raw"
export nnUNet_preprocessed="$1/preprocessed"
export RESULTS_FOLDER="$1/trained_models"

python3 train.py ${@:2}
