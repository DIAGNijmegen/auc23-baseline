#!/bin/bash
./build.sh $1

docker push doduo1.umcn.nl/universalclassifier/training:$1
