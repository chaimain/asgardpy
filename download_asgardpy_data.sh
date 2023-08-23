#!/bin/bash

#ASGARDPY_DATA=${ASGARDPY_DATA:-""}

OUT_ZIP="fermipy_crab_zipped.zip"

#curl \
#  "$ASGARDPY_DATA"  \
#  --output ${OUT_ZIP}

unzip \
  -n \
  ${OUT_ZIP} \
  -d $GAMMAPY_DATA"fermipy-crab/"
