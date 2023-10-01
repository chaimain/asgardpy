#!/bin/bash

set  -eu

GAMMAPY_DATA=${GAMMAPY_DATA:-""}

# Used "zip -r9 fermipy_crab_zipped.zip ./" to zip the files
OUT_ZIP="dev/fermipy_crab_zipped.zip"

echo $GAMMAPY_DATA" is the path to Gammapy datasets"

OUT_DIR=$GAMMAPY_DATA"fermipy-crab/"

mkdir -p $OUT_DIR

unzip \
  -u \
  ${OUT_ZIP} \
  -d $OUT_DIR
