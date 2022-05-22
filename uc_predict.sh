#!/bin/bash

if [ -z "$1" ];
do
  echo "Please pass as first argument to $0 a directory that contains ./trained_models/"
  exit
fi

if ! [ -d "$1/raw"];
do
  echo "Please make sure that $1/trained_models/ exists"
  exit
fi

export RESULTS_FOLDER="$1/trained_models"

python3 uc_predict.py ${@:2}
