#!/bin/bash

ASGARDPY_DATA=${ASGARDPY_DATA:-""}
OUT_ZIP="asgardpy_data.zip"

curl \
  "$ASGARDPY_DATA"  \
  --output ${OUT_ZIP}

unzip \
  ${OUT_ZIP} \
  -d $GAMMAPY_DATA
  
