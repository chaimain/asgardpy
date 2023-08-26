#!/bin/bash

set  -eu

ASGARDPY_DATA=${ASGARDPY_DATA:-""}
GAMMAPY_DATA=${GAMMAPY_DATA:-""}

OUT_ZIP="fermipy_crab_zipped.zip"

echo $GAMMAPY_DATA" is the path to Gammapy datasets"

OUT_DIR=$GAMMAPY_DATA"fermipy-crab/"

mkdir -p $OUT_DIR

unzip \
  -u \
  ${OUT_ZIP} \
  -d $OUT_DIR
