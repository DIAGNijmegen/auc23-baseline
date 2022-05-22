#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build --tag=doduo1.umcn.nl/universalclassifier/training:$1 "$SCRIPTPATH"
