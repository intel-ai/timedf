#!/bin/bash -eu
mkdir -p ${RESULTS_DIR}
chmod 0777 ${RESULTS_DIR}

USER_ID="$(id -u):$(id -g)"

docker run \
  -it \
  -v ${DATASETS_ROOT}:/datasets:ro  \
  -v ${RESULTS_DIR}:/results \
  --user ${USER_ID} \
  modin-project/benchmarks-reproduce:latest 
