docker run \
    -it \
  -v ${DATASETS_ROOT}:/datasets:ro  \
  -v ${RESULTS_DIR}:/results \
  modin-project/benchmarks-reproduce:latest \
  bash
