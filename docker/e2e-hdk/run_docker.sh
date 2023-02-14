#!/bin/bash -eu
if [ -z ${DATASETS_PATH+x} ]; then
    echo "Please, set the env variable \"DATASETS_PATH\"!";
    exit 1
else
    echo "DATSETS_ROOT=$DATASETS_PATH"
fi

mkdir -p ${RESULTS_DIR}
chmod 0777 ${RESULTS_DIR}

USER_ID="$(id -u):$(id -g)"

docker run \
  -it \
  -v ${DATASETS_PATH}:/datasets:ro  \
  -v ${RESULTS_DIR}:/results \
  --user ${USER_ID} \
  --env REPORT=$REPORT \
  modin-project/benchmarks-reproduce:latest 
