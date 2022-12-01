#!/bin/bash -eu
docker run \
  -v ${DATASETS_ROOT}:/datasets:ro  \
  -v ${RESULTS_DIR}:/results \
  modin-project/benchmarks-reproduce:latest
