#!/bin/bash

set  -eu

ASGARDPY_DATA=${ASGARDPY_DATA:-""}
GAMMAPY_DATA=${GAMMAPY_DATA:-""}

OUT_ZIP="fermipy_crab_zipped.zip"

echo "{$GAMMAPY_DATA} is the path to Gammapy datasets"

#curl \
#  "$ASGARDPY_DATA"  \
#  --output ${OUT_ZIP}

unzip \
  -u \
  -v \
  ${OUT_ZIP} \
  -exdir $GAMMAPY_DATA"fermipy-crab/"
