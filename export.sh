#!/usr/bin/env bash

./build.sh

docker save universalclassifier | gzip -c > universalclassifier.tar.gz
