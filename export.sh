#!/usr/bin/env bash

./build.sh $1

docker save doduo1.umcn.nl/universalclassifier/training:$1 | gzip -c > universalclassifier_training_v$1.tar.gz
